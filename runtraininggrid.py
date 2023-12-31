import os
import time
import datetime
import torch
import wandb
import sys
sys.path.append(".")
from source.trainer import TrainerConfig, Trainer
from source.transformer import TransformerConfig, Transformer
from source.tokenizer import Tokenizer
from source.dataset import DatasetConfig, Dataset


def grid_search():
    num_epochs = 500
    info = "info" in sys.argv
    train(n_layer=2, n_head=2, n_embd=256, num_epochs=num_epochs, info=info)
    train(n_layer=4, n_head=4, n_embd=256, num_epochs=num_epochs, info=info)
    train(n_layer=6, n_head=8, n_embd=256, num_epochs=num_epochs, info=info)
    train(n_layer=8, n_head=8, n_embd=256, num_epochs=num_epochs, info=info)


def train(n_layer=2, n_head=2, n_embd=128, dropout=0, bottleneck_depth=5, num_epochs=100, batch_size=128, info=False):
    # Get the timestamp as YYYYMMDD-HHMM.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Create the dataset config and dataset.
    dataset_config = DatasetConfig(
        dataset_path = "data/jsfakes4bars/generation"
    )
    dataset = Dataset(
        dataset_config,
    )
    print(dataset)

    # Create the training config.
    trainer_config = TrainerConfig(
        device="auto",
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Create the model config.
    model_config = TransformerConfig(
        vocab_size = 128,
        n_layer = n_layer,
        n_head = n_head,
        n_embd = n_embd,
        dropout = dropout,
        bias = False,
        block_size = 384,
        bottleneck = "variational", # "simple" or "variational" or "none"
        bottleneck_depth = bottleneck_depth,
    )

    # Set the output directory.
    trainer_config.out_dir = os.path.join(trainer_config.out_dir, f"transformer_{model_config.bottleneck}_{timestamp}")

    # Set the model config.
    trainer_config.wandb_log = True
    trainer_config.wandb_project = "bottleneck-transformers-20231001"
    trainer_config.wandb_run_name = f"transformer_{model_config.bottleneck}_{timestamp}"

    # Create the model.
    model = Transformer(model_config)

    # Create the trainer.
    trainer = Trainer(trainer_config)

    # Train or info.
    if not info:
        trainer.train(model, dataset)
    else:
        trainer.info(model)


if __name__ == "__main__":
    grid_search()