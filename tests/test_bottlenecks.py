import torch
import numpy as np
import sys
sys.path.append(".")
from source.bottlenecks import SimpleBottleneck, VariationalBottleneck

def main():

    #test_simple_bottleneck()
    test_variational_bottleneck()

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
    print(f"y.shape: {y.shape} ({np.prod(y.shape)} numbers)")

    # Assert the shapes are the same.
    assert y.shape == x.shape


def test_variational_bottleneck():

    # Create the bottleneck.
    block_size = 512
    n_embd = 128
    depth = 4
    bottleneck = VariationalBottleneck(block_size, n_embd, depth)
    
    # Create a random sample.
    x = torch.rand(1, block_size, n_embd)
    print(f"x.shape: {x.shape} ({np.prod(x.shape)} numbers)")
    print("")
    
    # Encode the sample.
    mu, logvar = bottleneck.encode(x)
    print(f"mu.shape: {mu.shape} ({np.prod(mu.shape)} numbers)")
    print(f"logvar.shape: {logvar.shape} ({np.prod(logvar.shape)} numbers)")
    print("")

    z = bottleneck.reparameterize(mu, logvar)
    print(f"z.shape: {z.shape} ({np.prod(z.shape)} numbers)")
    print("")

    # Decode the sample.
    y = bottleneck.decode(z)
    print(f"y.shape: {y.shape} ({np.prod(y.shape)} numbers)")
    print("")

    # Test the forward pass with passing mask.
    padding_mask = torch.ones(1, block_size, dtype=torch.float32)
    padding_mask[:, 64:] = 0
    print(f"padding_mask.shape: {padding_mask.shape} ({np.prod(padding_mask.shape)} numbers)")
    print(f"padding_mask: {padding_mask}")
    y, loss = bottleneck(x, return_loss=True, padding_mask=padding_mask)
    print(f"y.shape: {y.shape} ({np.prod(y.shape)} numbers)")
    print(f"loss: {loss}")

    # Assert the shapes are the same.
    assert y.shape == x.shape

if __name__ == '__main__':
    main()