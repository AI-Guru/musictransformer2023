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
    # Get the timestamp as YYYYMMDD-HHMM.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Create the training config.
    trainer_config = TrainerConfig(
        device="auto"
    )

    # Create the dataset config and dataset.
    dataset_config = DatasetConfig(
        dataset_path = "data/jsfakes4bars/generation"
        token_dropout = False # No data augmentation.

    )
    dataset = Dataset(
        dataset_config,
    )
    print(dataset)

    # Create the model config.
    model_config = TransformerConfig(
        vocab_size = 128,
        n_layer = 8,
        n_head = 8,
        n_embd = 512,
        dropout = 0.0,
        bias = False,
        block_size = 384,
        bottleneck = "variational", # "simple" or "variational" or "none"
        bottleneck_depth = 5
    )

    # Set the output directory.
    trainer_config.out_dir = os.path.join(trainer_config.out_dir, f"transformer_{model_config.bottleneck}_{timestamp}")

    # Set the model config.
    trainer_config.wandb_log = True
    trainer_config.wandb_project = "bottleneck-transformers-DEV"
    trainer_config.wandb_run_name = f"transformer_{model_config.bottleneck}_{timestamp}"
    trainer_config.wandb_group = "transformer"
    bottleneck_loss_coefficient = 1.0 # The beta of the variational bottleneck loss. Start value.
    bottleneck_loss_coefficient_max = 1.0 # End value.

    # Create the model.
    model = Transformer(model_config)

    # Create the trainer.
    trainer = Trainer(trainer_config)
    trainer.train(model, dataset)

if __name__ == "__main__":
    train()