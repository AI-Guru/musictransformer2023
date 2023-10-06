import math
import inspect
from dataclasses import dataclass, replace
import os
import json

import torch
import torch.nn as nn
from torch.nn import functional as F

import sys
sys.path.append("..")

from source.layers import (
    EncoderBlock,
    DecoderBlock,
    LayerNorm,
)
from source.bottlenecks import (
    SimpleBottleneck,
    VariationalBottleneck,
)
from source.tokenizer import Tokenizer

@dataclass
class TransformerConfig:
    block_size: int = 256 #1024
    vocab_size: int = 400 #50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4 #12
    n_head: int = 4# 12
    n_embd: int = 128 #768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bottleneck: str = "none"
    #bottleneck_loss_coef: float = 100.0
    bottleneck_depth: int = 4
    flash_attention: bool = False
    

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # The encoder part.
        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # The decoder part.
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # The bottleneck part.
        if config.bottleneck == "none":
            self.bottleneck = None
        elif config.bottleneck == "simple":
            self.bottleneck = SimpleBottleneck(config.block_size, config.n_embd)
        elif config.bottleneck == "variational":
            self.bottleneck = VariationalBottleneck(config.block_size, config.n_embd, config.bottleneck_depth)
        else: 
            raise NotImplementedError(f"Bottleneck {config.bottleneck} not implemented yet.")

        # The head part.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # https://paperswithcode.com/method/weight-tying
        self.encoder.wte.weight = self.lm_head.weight 
        self.decoder.wte.weight = self.lm_head.weight 

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
            n_params -= self.encoder.wpe.weight.numel()
            n_params -= self.decoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, encoder_ids, decoder_ids, target_ids=None, padding_mask=None):
        device = encoder_ids.device
        b, t = encoder_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Forward the encoder.
        tok_emb_encoder = self.encoder.wte(encoder_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb_encoder = self.encoder.wpe(pos) # position embeddings of shape (t, n_embd)
        x_encoder = self.encoder.drop(tok_emb_encoder + pos_emb_encoder)
        for encoder_block in self.encoder.h:
            x_encoder = encoder_block(x_encoder)
        x_encoder = self.encoder.ln_f(x_encoder)
        
        # Forward the bottleneck if it exists.
        bottleneck_loss = 0.0
        if self.bottleneck is not None:
            x_encoder, bottleneck_loss = self.bottleneck(
                x_encoder,
                return_loss=True,
                padding_mask=padding_mask,
            ) 
            #bottleneck_loss = bottleneck_loss * self.config.bottleneck_loss_coef

        # Forward the decoder.
        tok_emb_decoder = self.decoder.wte(decoder_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb_decoder = self.decoder.wpe(pos) # position embeddings of shape (t, n_embd)
        x_decoder = self.decoder.drop(tok_emb_decoder + pos_emb_decoder)
        for decoder_block in self.decoder.h:
            x_decoder = decoder_block(x_decoder, x_encoder)
        x_decoder = self.decoder.ln_f(x_decoder)

        if target_ids is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x_decoder)
            reconstruction_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-1)
            # Apply the padding mask.
            if padding_mask is not None:
                reconstruction_loss = reconstruction_loss * padding_mask.view(-1)
                reconstruction_loss = reconstruction_loss.sum() / padding_mask.sum()
            #loss = reconstruction_loss + bottleneck_loss
            return logits, reconstruction_loss, bottleneck_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x_decoder[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
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
        tok_emb_encoder = self.encoder.wte(encoder_ids)
        pos_emb_encoder = self.encoder.wpe(pos)
        x_encoder = self.encoder.drop(tok_emb_encoder + pos_emb_encoder)

        for encoder_block in self.encoder.h:
            x_encoder = encoder_block(x_encoder)

        x_encoder = self.encoder.ln_f(x_encoder)

        return x_encoder


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

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

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
    def generate(self, decoder_ids, encoder_ids=None, bottleneck_condition=None, max_new_tokens=128, end_token_id=None, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # Ensure that idx is a tensor.
        if not isinstance(decoder_ids, torch.LongTensor):
            decoder_ids = torch.LongTensor(decoder_ids)

        # Ensure that the first dimension is batch size.
        if decoder_ids.dim() == 1:
            decoder_ids = decoder_ids.unsqueeze(0)

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

        # Use the encoder ids if they exist.
        elif encoder_ids is not None:
            # Make sure that the encoder ids are a tensor.
            if not isinstance(encoder_ids, torch.LongTensor):
                encoder_ids = torch.LongTensor(encoder_ids)

            # Make sure that the first dimension is batch size.
            if encoder_ids.dim() == 1:
                encoder_ids = encoder_ids.unsqueeze(0)

            # Forward the encoder.
            encoder_x = self.encode(encoder_ids)

        # Generate tokens.
        for _ in range(max_new_tokens):

            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = decoder_ids if decoder_ids.size(1) <= self.config.block_size else decoder_ids[:, -self.config.block_size:]

            # forward the model to get the logits for the index in the sequence
            #logits, _ = self(idx_cond)
            logits = self.forward_encoder(encoder_x, idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            decoder_ids = torch.cat((decoder_ids, idx_next), dim=1)

            # If the latest token emitted is an <END> token, then we're done.
            if end_token_id is not None and idx_next.squeeze().item() == end_token_id:
                break

        return decoder_ids


    #@classmethod
    def load(check_point_path):
        if not os.path.exists(check_point_path):
            raise Exception(f"Could not find checkpoint at {check_point_path}")
        # Load the checkpoint.
        checkpoint = torch.load(check_point_path, map_location="cpu")

        # Print the keys of the checkpoint.
        #print("Checkpoint keys:")
        #for key in checkpoint.keys():
        #    print(key)

        # Load the config.
        model_config_dict = checkpoint["model_config"]
        model_config = TransformerConfig(**model_config_dict)

        # Create the model.
        model = Transformer(model_config)

        # Get the keys of the state dict. Sort them and make a list.
        state_dict_keys = model.state_dict().keys()
        state_dict_keys = sorted(state_dict_keys)

        # Get the keys of the loaded state dict. Sort them and make a list.
        loaded_state_dict_keys = checkpoint["model"].keys()
        loaded_state_dict_keys = sorted(loaded_state_dict_keys)

        # Get the keys that both lists have in common. XOR.
        #common_keys = set(state_dict_keys).intersection(loaded_state_dict_keys)
        #print(f"common_keys: {len(common_keys)}")

        # Get the keys that they don't have in common. XOR.
        missing_keys = set(state_dict_keys).symmetric_difference(loaded_state_dict_keys)
        if len(missing_keys) > 0:
            raise Exception(f"Missing {len(missing_keys)} keys: {missing_keys}")
        

        # Both lists should have the same length.
        assert len(state_dict_keys) == len(loaded_state_dict_keys), f"Length of state_dict_keys ({len(state_dict_keys)}) does not match length of loaded_state_dict_keys ({len(loaded_state_dict_keys)})"

        # Load the state dict into the model.
        model.load_state_dict(checkpoint["model"])

        # Return the model.
        return model
    
    def get_bottleneck_shape(self):
        # Raise an exception if there is no bottleneck.
        if self.bottleneck is None:
            raise Exception("No bottleneck!")
        
        # Return the shape.
        return self.bottleneck.get_shape()
