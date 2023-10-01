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


def grid_search():
    train(n_layer=2, n_head=2, n_embd=256)
    train(n_layer=4, n_head=4, n_embd=256)
    train(n_layer=6, n_head=8, n_embd=256)
    train(n_layer=8, n_head=8, n_embd=256)


def train(n_layer=2, n_head=2, n_embd=128, dropout=0, bottleneck_depth=5):
    # Get the timestamp as YYYYMMDD-HHMM.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Create the training config.
    trainer_config = TrainerConfig()
    trainer_config.device = "auto"

    # Create the model config.
    model_config = TransformerConfig()
    model_config.vocab_size = 118
    model_config.n_layer = n_layer
    model_config.n_head = n_head
    model_config.n_embd = n_embd
    model_config.dropout = dropout
    model_config.bias = False
    model_config.block_size = 384
    model_config.bottleneck = "variational" # "simple" or "variational" or "none"
    bottleneck_loss_coef = 100.0
    model_config.bottleneck_depth = bottleneck_depth

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
    trainer.train(model)


if __name__ == "__main__":
    grid_search()