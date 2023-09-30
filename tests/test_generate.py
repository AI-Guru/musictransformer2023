
import torch
import numpy as np
import sys
sys.path.append(".")
from source.transformer import TransformerConfig, Transformer

def main():

    #test_simple_bottleneck()
    test_generate_with_bottleneck_condition()
    test_generate_with_encoder_ids()


def test_generate_with_bottleneck_condition():

    # Create model.
    config = TransformerConfig()
    config.bottleneck = "variational"
    model = Transformer(config)

    # Sample from a normal distribution.
    bottleneck_shape = model.get_bottleneck_shape()
    bottleneck_z = torch.randn(1, *bottleneck_shape)
    print(f"bottleneck shape: {bottleneck_shape}, numbers {np.prod(bottleneck_shape)}")

    # Generate.
    result_ids = model.generate(
        decoder_ids=[0],
        max_new_tokens=10,
        bottleneck_condition=bottleneck_z, 
        temperature=0.2,
        top_k=None
    )[0]
    print(f"Result ids: {result_ids}")

    
def test_generate_with_encoder_ids():
    # Create model.
    config = TransformerConfig()
    config.bottleneck = "variational"
    model = Transformer(config)

    # Generate.
    result_ids = model.generate(
        decoder_ids=[0],
        encoder_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        max_new_tokens=10,
        temperature=0.2,
        top_k=None
    )[0]
    print(f"Result ids: {result_ids}")


if __name__ == '__main__':
    main()