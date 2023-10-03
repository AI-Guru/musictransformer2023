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
    )
    dataset = Dataset(
        dataset_config,
    )
    print(dataset)

    # Create the model config.
    model_config = TransformerConfig()
    model_config.vocab_size = 128
    model_config.n_layer = 8
    model_config.n_head = 8
    model_config.n_embd = 512
    model_config.dropout = 0.0
    model_config.bias = False
    model_config.block_size = 384
    model_config.bottleneck = "variational" # "simple" or "variational" or "none"
    model_config.bottleneck_depth = 5

    # Set the output directory.
    trainer_config.out_dir = os.path.join(trainer_config.out_dir, f"transformer_{model_config.bottleneck}_{timestamp}")

    # Set the model config.
    trainer_config.wandb_log = True
    trainer_config.wandb_project = "bottleneck-transformers-DEV"
    trainer_config.wandb_run_name = f"transformer_{model_config.bottleneck}_{timestamp}"

    # Create the model.
    model = Transformer(model_config)

    # Create the trainer.
    trainer = Trainer(trainer_config)
    trainer.train(model, dataset)

if __name__ == "__main__":
    train()