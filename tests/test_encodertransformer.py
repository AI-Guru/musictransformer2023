import torch
import numpy as np
import sys
sys.path.append(".")
from source.encodertransformer import EncoderTransformerConfig, EncoderTransformer

def main():

    test()

def test():

    print("Testing Transformer...")
    
    transformer_config = EncoderTransformerConfig(
        bottleneck="variational",
        #bottleneck_depth=5,
    )
    print(transformer_config)
    transformer = EncoderTransformer(transformer_config)
    print(transformer)

    # Do a forward pass.
    # x is a sequence of integers.
    sequence_length = transformer_config.block_size
    vocab_size = transformer_config.vocab_size
    batch_size = 128
    encoder_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    print(f"encoder_ids.shape: {encoder_ids.shape}")
    print(f"encoder_ids: {encoder_ids}")
    target_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    print(f"target_ids.shape: {target_ids.shape}")
    print(f"target_ids: {target_ids}")
    padding_mask = torch.ones((batch_size, sequence_length))
    padding_mask[:, sequence_length//2:] = 0
    print(f"padding_mask.shape: {padding_mask.shape}")
    print(f"padding_mask: {padding_mask}")

    # y is a sequence of logits.
    logits, reconstruction_loss, bottleneck_loss = transformer(
        encoder_ids,
        target_ids=target_ids,
        padding_mask=padding_mask,
    )
    print(f"logits.shape: {logits.shape}")
    print(f"logits: {logits}")
    print(f"reconstruction_loss: {reconstruction_loss}")
    print(f"bottleneck_loss: {bottleneck_loss}")
    print("")

    print(transformer.summary())


if __name__ == '__main__':
    main()