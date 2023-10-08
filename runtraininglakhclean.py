import os
import time
import datetime
import sys
sys.path.append(".")
from source.trainer import TrainerConfig, Trainer
from source.transformer import TransformerConfig, Transformer
from source.tokenizer import Tokenizer
from source.dataset import DatasetConfig, Dataset


def train():

    test = "test" in sys.argv

    # Get the timestamp as YYYYMMDD-HHMM.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Create the Training config.
    trainer_config = TrainerConfig(

        # General settings.
        out_dir=f"output/lakhclean/transformer_{timestamp}",
        batch_size=128,
        num_epochs=5,
        device="auto",
        
        # Bottleneck config.
        bottleneck_loss_coefficient=1.0,
        bottleneck_loss_coefficient_max=1.0,
        
        # Optimizer settings.
        learning_rate=6e-4, # Max learning rate.
        max_iters=50_000,   # Total number of training iterations.
        weight_decay=1e-1,  # Weight decay.
        beta1=0.9,          # Beta1 for Adam.
        beta2=0.95,         # Beta2 for Adam.
        grad_clip=1.0,      # Clip gradients at this value, or disable if == 0.0.

        # Learning rate decay settings.
        decay_lr=True,          # Whether to decay the learning rate.
        warmup_iters=1_000,     # How many steps to warm up for.
        lr_decay_iters=50_000,  # Should be ~= max_iters per Chinchilla.
        min_lr=6e-5,            # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla.

        # Wandb config.
        wandb_log=True,
        wandb_project="transformer-vae-lakhclean",
        wandb_run_name=f"transformer_{timestamp}",
        
        # When to evaluate.
        eval_every=1000,
        eval_mode="steps",

        # When to log.
        log_every=1000,
        log_mode="steps",

        # When to save.
        save_every=1000,
        save_mode= "steps",
    )

    # Create the trainer.
    trainer = Trainer(trainer_config)

    # Create the dataset config and dataset.
    dataset_config = DatasetConfig(
        dataset_path = "data/lakhclean_mmmtrack_1bars_vae/generation",
        token_dropout = False # No data augmentation.

    )
    dataset = Dataset(
        dataset_config,
    )
    print(dataset)

    # Create the model config.
    model_config = TransformerConfig(
        vocab_size = 320,
        n_layer = 3,
        n_head = 8,
        n_embd = 512,
        dropout = 0.0,
        bias = False,
        block_size = 512,
        bottleneck = "variational", # "simple" or "variational" or "none"
        bottleneck_depth = 3
    )

    # Create the model.
    model = Transformer(model_config)

    # Train.
    trainer.train(model, dataset)



if __name__ == "__main__":
    train()