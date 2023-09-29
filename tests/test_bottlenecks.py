import torch
import numpy as np
import sys
sys.path.append(".")
from source.bottlenecks import SimpleBottleneck

def main():

    test_simple_bottleneck()

def test_simple_bottleneck():

    # Create the bottleneck.
    block_size = 384
    n_embd = 128
    bottleneck = SimpleBottleneck(block_size, n_embd)
    
    # Create a random sample.
    x = torch.rand(1, block_size, n_embd)
    print(f"x.shape: {x.shape}")
    print(f"  {np.prod(x.shape)} numbers")
    
    # Encode the sample.
    z = bottleneck.encode(x)
    print(f"z.shape: {z.shape}")
    print(f"  {np.prod(z.shape)} numbers")

    # Decode the sample.
    y = bottleneck.decode(z)
    print(f"y.shape: {y.shape}")
    print(f"  {np.prod(y.shape)} numbers")

    # Assert the shapes are the same.
    assert y.shape == x.shape


if __name__ == '__main__':
    main()