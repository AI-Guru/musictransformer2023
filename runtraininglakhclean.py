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
        out_dir=f"output/lakhclean/transformer_{timestamp}",
        batch_size=160,
        num_epochs=1,
        device="auto",
        
        # Bottleneck config.
        bottleneck_loss_coefficient=1.0,
        bottleneck_loss_coefficient_max=1.0,
        
        # Wandb config.
        wandb_log=True,
        wandb_project="transformer-vae-lakhclean",
        wandb_run_name=f"transformer_{timestamp}",
        
        # When to evaluate.
        eval_every=500,
        eval_mode="steps",

        # When to log.
        log_every=500,
        log_mode="steps",

        # When to save.
        save_every=500,
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
        block_size = 786,
        bottleneck = "variational", # "simple" or "variational" or "none"
        bottleneck_depth = 5
    )

    # Create the model.
    model = Transformer(model_config)

    # Train.
    trainer.train(model, dataset)



if __name__ == "__main__":
    train()