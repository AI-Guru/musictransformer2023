import os
import torch
import numpy as np
import tempfile
import glob
import sys
sys.path.append(".")
from source.trainer import TrainerConfig, Trainer
from source.transformer import TransformerConfig, Transformer
from source.encodertransformer import EncoderTransformerConfig, EncoderTransformer

def main():

    #test_load_existing()

    test_load_save(TransformerConfig, Transformer)
    test_load_save(EncoderTransformerConfig, EncoderTransformer)


def test_load_existing():
    # Find all the .pt files in the models directory.
    # Load each one.

    # Search recursively for all .pt files in the models directory.
    checkpoint_paths = glob.glob("models/**/*.pt", recursive=True)
    print(f"Found {len(checkpoint_paths)} checkpoints")

    for checkpoint_path in checkpoint_paths:
        print(f"Loading {checkpoint_path}")
        transformer = Transformer.load(checkpoint_path)
        print("Loaded transformer")


def test_load_save(config_class, model_class):

    # Create a transformer. 
    # Store it to a file in a temporary directory.
    # Load it back from the file.
    with tempfile.TemporaryDirectory() as tempdir:
        config = config_class()
        transformer = model_class(config)
    
        # Save the model via the trainer.
        trainer_config = TrainerConfig()
        trainer = Trainer(trainer_config)
        checkpoint_name = "checkpoint"
        checkpoint_path = trainer.save_checkpoint(model=transformer, optimizer=None, step=None, epoch=None, checkpoint_name=checkpoint_name)
        print(f"checkpoint_path: {checkpoint_path}")

        # Load the model via the transformer.
        transformer = model_class.load(checkpoint_path)
        print("Loaded transformer")


def test_variational_bottleneck():

    # Create the bottleneck.
    block_size = 384
    n_embd = 128
    depth = 5
    bottleneck = VariationalBottleneck(block_size, n_embd, depth)
    print(f"bottleneck: {bottleneck}")
    print("")
    
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

    # Assert the shapes are the same.
    assert y.shape == x.shape

if __name__ == '__main__':
    main()