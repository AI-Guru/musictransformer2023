import os
import time
import datetime
import sys
sys.path.append(".")
from source.trainer import TrainerConfig, Trainer
from source.encodertransformer import EncoderTransformerConfig, EncoderTransformer
from source.bottlenecks import (
    VariationalCNNBottleneck,
    VariationalLinear2DBottleneck
)
from source.dataset import DatasetConfig, Dataset


def train():

    test = "test" in sys.argv

    # Get the timestamp as YYYYMMDD-HHMM.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Create the model config.
    model_config = EncoderTransformerConfig(
        vocab_size = 320,
        n_layer = 2,
        n_head = 4,
        n_embd = 128,
        dropout = 0.0,
        bias = False,
        block_size = 512,

        weight_sharing = True,
        
        # Uses a CNN to extract features from the 2D samples.
        bottleneck = "CNNBottleneck",
        #bottleneck = "VariationalCNNBottleneck",
        #bottleneck_channels_list=[128, 256],
        
        # Keeps the 2D samples.
        #bottleneck = "Linear2DBottleneck",
        #bottleneck = "VariationalLinear2DBottleneck",
        #bottleneck_channels_list=[64, 16, 2],

        # Reshapes the 2D samples into 1D samples.
        #bottleneck = "Linear1DBottleneck",
        #bottleneck = "VariationalLinear1DBottleneck",
        #bottleneck_channels_list=[1024, 128],

        #bottleneck = "VariationalLinear1DBottleneck",
        #bottleneck_channels_list=[1024, 128],

    )

    # Create the model.
    model = EncoderTransformer(model_config)
    print(model.summary())
    if "summary" in sys.argv:
        return

    # Create the dataset config and dataset.
    dataset_config = DatasetConfig(
        dataset_path = "data/lakhclean_mmmtrack_1bars_vae/generation",
        token_dropout = False # No data augmentation.

    )
    dataset = Dataset(
        dataset_config,
    )
    print(dataset)

    # Create the Training config.
    trainer_config = TrainerConfig(

        # General settings.
        out_dir=f"output_debug/lakhclean/encodertransformer_{timestamp}",
        batch_size=128,
        num_epochs=5,
        device="auto",
        
        # Bottleneck config.
        bottleneck_loss_coefficient=0.1,
        bottleneck_loss_coefficient_max=0.1,
        bottleneck_loss_iterations=15_000,
        
        # Optimizer settings.
        learning_rate=6e-5, # Max learning rate.
        min_lr=6e-6,            # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla.
        max_iters=50_000,   # Total number of training iterations.
        weight_decay=1e-1,  # Weight decay.
        beta1=0.9,          # Beta1 for Adam.
        beta2=0.95,         # Beta2 for Adam.
        grad_clip=1.0,      # Clip gradients at this value, or disable if == 0.0.

        # Learning rate decay settings.
        decay_lr=True,          # Whether to decay the learning rate.
        warmup_iters=1_000,     # How many steps to warm up for.
        lr_decay_iters=50_000,  # Should be ~= max_iters per Chinchilla.

        # Wandb config.
        wandb_log=True,
        wandb_project="encodertransformer-vae-lakhclean",
        wandb_run_name=f"encodertransformer_{timestamp}",
        
        # When to evaluate.
        eval_every=500,
        eval_mode="steps",

        # When to log.
        log_every=500,
        log_mode="steps",

        # When to save.
        save_every=500,
        save_mode= "steps",

        # Debugging.
        find_not_updated_layers=True,
        max_eval_steps=500,
        log_grad_norm=True,
    )

    # Create the trainer.
    trainer = Trainer(trainer_config)

    # Train.
    trainer.train(model, dataset)



if __name__ == "__main__":
    train()