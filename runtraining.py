import os
import time
import datetime
import math
import pickle
from contextlib import nullcontext
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from datasets import load_dataset # huggingface datasets

import sys
sys.path.append(".")

#from model import GPTConfig, GPT
from source.trainer import TrainerConfig, Trainer
from source.transformer import TransformerConfig, Transformer
from source.tokenizer import Tokenizer

from dataclasses import dataclass, asdict


# Get the timestamp as YYYYMMDD-HHMM.
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

# Create the training config.
trainer_config = TrainerConfig()
trainer_config.device = "auto"


# Create the model config.
model_config = TransformerConfig()
model_config.vocab_size = 118
model_config.n_layer = 8
model_config.n_head = 8
model_config.n_embd = 512
model_config.dropout = 0.0
model_config.bias = False
model_config.block_size = 384
model_config.bottleneck = "variational" # "simple" or "variational" or "none"
bottleneck_loss_coef = 100.0
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
trainer.train(model)