import math
import inspect
from dataclasses import dataclass, field
from typing import List
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import sys
sys.path.append("..")

from source.layers import (
    EncoderBlock,
    LayerNorm,
)
from source.bottlenecks import (
    BottleneckFactory
)
from source.basetransformer import BaseTransformerConfig, BaseTransformer

@dataclass
class EncoderTransformerConfig(BaseTransformerConfig):
    pass
    

class EncoderTransformer(BaseTransformer):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        
        # Store the config.
        self.config = config

        # Set the family.
        self.family = "encoder"

        # Token an position embeddings.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # The encoder part.
        self.encoder = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Is there a class with the name config.bottleneck in source.bottlenecks?
        # Use reflection to get the class.
        # The bottleneck part.
        if config.bottleneck == "none":
            self.bottleneck = None
        else:
            the_class = BottleneckFactory.get_class(config.bottleneck)
            if the_class is None:
                raise Exception(f"Could not find class {config.bottleneck} in source.bottlenecks.")
            else:
                self.bottleneck = the_class(config)

        # The decoder part.
        self.decoder = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # The head part.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # https://paperswithcode.com/method/weight-tying
        if config.weight_sharing:
            self.wte.weight = self.lm_head.weight 

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, encoder_ids, target_ids=None, padding_mask=None):

        # padding mask should be float.
        assert padding_mask is None or padding_mask.dtype == torch.float32, f"padding_mask.dtype is {padding_mask.dtype}"

        device = encoder_ids.device
        b, t = encoder_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Forward the encoder.
        tok_emb_encoder = self.wte(encoder_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb_encoder = self.wpe(pos) # position embeddings of shape (t, n_embd)
        x_encoder = self.encoder.drop(tok_emb_encoder + pos_emb_encoder)
        for encoder_block in self.encoder.h:
            x_encoder = encoder_block(x_encoder)
        x_encoder = self.encoder.ln_f(x_encoder)
        
        # Forward the bottleneck if it exists.
        bottleneck_loss = None
        if self.bottleneck is not None:
            x_encoder, bottleneck_loss = self.bottleneck(
                x_encoder,
                return_loss=True,
                padding_mask=padding_mask,
            ) 
            assert isinstance(bottleneck_loss, torch.Tensor) or bottleneck_loss is None, f"bottleneck_loss is {type(bottleneck_loss)}"

        # Add the position embeddings.
        x_encoder = x_encoder + pos_emb_encoder

        # Forward the decoder.
        x_decoder = self.decoder.drop(x_encoder)
        for decoder_block in self.decoder.h:
            x_decoder = decoder_block(x_decoder)
        x_decoder = self.decoder.ln_f(x_decoder)

        # For training.
        if target_ids is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x_decoder)
            reconstruction_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-1)
            assert isinstance(reconstruction_loss, torch.Tensor), f"reconstruction_loss is {type(reconstruction_loss)}"
            # Apply the padding mask.
            if padding_mask is not None:
                reconstruction_loss = reconstruction_loss * padding_mask.view(-1)
                reconstruction_loss = reconstruction_loss.sum() / padding_mask.sum()
            #loss = reconstruction_loss + bottleneck_loss
            return logits, reconstruction_loss, bottleneck_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x_decoder[:, [-1], :]) # note: using list [-1] to preserve the time dim
            return logits


    def forward_encoder(self, x_encoder, decoder_ids):
        # Assume that we already have the embedding of the encoder. 
        # Only forward the decoder.

        b, t = decoder_ids.size()
        device = decoder_ids.device
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Forward the decoder.
        tok_emb_decoder = self.decoder.wte(decoder_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb_decoder = self.decoder.wpe(pos) # position embeddings of shape (t, n_embd)

        x_decoder = tok_emb_decoder + pos_emb_decoder

        # Ensure that x_decoder has the same shape as encoder_x. Encoder x is longer. Truncate it.
        #if x_decoder.shape[1] < x_encoder.shape[1]:
        #    x_encoder = x_encoder[:, :x_decoder.shape[1], :]

        x_decoder = self.decoder.drop(x_decoder)
        for decoder_block in self.decoder.h:
            x_decoder = decoder_block(x_decoder, x_encoder)

        x_decoder = self.decoder.ln_f(x_decoder)

        # Get the logits.
        logits = self.lm_head(x_decoder[:, [-1], :])
        return logits

    def encode(self, encoder_ids):

        b, t = encoder_ids.size()

        device = encoder_ids.device
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Forward the encoder.
        tok_emb_encoder = self.wte(encoder_ids)
        pos_emb_encoder = self.wpe(pos)
        x_encoder = self.encoder.drop(tok_emb_encoder + pos_emb_encoder)

        for encoder_block in self.encoder.h:
            x_encoder = encoder_block(x_encoder)

        x_encoder = self.encoder.ln_f(x_encoder)

        return x_encoder
    
    def decode(self, z):

        # Forward the decoder.
        x_decoder = self.decoder.drop(z)
        for decoder_block in self.decoder.h:
            x_decoder = decoder_block(x_decoder)
        x_decoder = self.decoder.ln_f(x_decoder)

        # Get the logits.
        logits = self.lm_head(x_decoder)
        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


    @torch.no_grad()
    def generate(self, encoder_ids=None, bottleneck_condition=None, temperature=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # Encoder ids and bottleneck condition cannot both be specified.
        if encoder_ids is not None and bottleneck_condition is not None:
            raise Exception("Cannot specify both encoder_ids and bottleneck_condition.")
        
        # Encoder ids and bottleneck condition cannot both be unspecified.
        elif encoder_ids is None and bottleneck_condition is None:
            raise Exception("Must specify either encoder_ids or bottleneck_condition.")
        
        # Use the bottleneck if it exists.
        elif bottleneck_condition is not None:
            assert self.bottleneck is not None, "Cannot generate without a bottleneck."
            encoder_x = self.bottleneck.decode(bottleneck_condition)
            print(f"encoder_x.shape: {encoder_x.shape}")

        # Use the encoder ids if they exist.
        elif encoder_ids is not None:
            # Make sure that the encoder ids are a tensor.
            if not isinstance(encoder_ids, torch.LongTensor):
                encoder_ids = torch.LongTensor(encoder_ids)

            print(f"encoder_ids.shape: {encoder_ids.shape}")

            # Make sure that the first dimension is batch size.
            if encoder_ids.dim() == 1:
                encoder_ids = encoder_ids.unsqueeze(0)

            # Forward the encoder.
            encoder_x = self.encode(encoder_ids)
            print(f"encoder_x.shape: {encoder_x.shape}")

            # If the bottleneck exists, use it.
            if self.bottleneck is not None:
                encoder_x = self.bottleneck(encoder_x)
                print(f"encoder_x.shape after bottleneck: {encoder_x.shape}")

        # Now decode. 
        logits = self.decode(encoder_x)
        assert len(logits.shape) == 3, f"len(logits.shape) is {len(logits.shape)}"
        print(f"logits.shape: {logits.shape}")

        # Apply the temperature.
        if temperature != 0.0:
            logits = logits / temperature

        # Get the probabilities.
        probs = F.softmax(logits, dim=-1)
        assert len(probs.shape) == 3, f"len(probs.shape) is {len(probs.shape)}"
        print(f"probs.shape: {probs.shape}")

        # Sample from the distribution.
        # Keep in mind that the first dimension is batch size and the second dimension is sequence length.
        # Sample from the last dimension.
        # Return shape is (batch_size, sequence_length, 1).
        indices = []
        for batch_index in range(probs.shape[0]):
            batch_indices = []
            for sequence_index in range(probs.shape[1]):
                sample = torch.multinomial(probs[batch_index, sequence_index, :], num_samples=1)[0]
                batch_indices.append(sample)
            batch_indices = torch.stack(batch_indices, dim=0)
            indices.append(batch_indices)
        indices = torch.stack(indices, dim=0)
        print(f"indices.shape: {indices.shape}")
        assert len(indices.shape) == 2, f"len(idx_next.shape) is {len(indices.shape)} shape is {indices.shape}"

        # Return the indices.
        return indices


    #@classmethod
    def load(check_point_path):

        return BaseTransformer.load_generic(check_point_path, EncoderTransformerConfig, EncoderTransformer)
