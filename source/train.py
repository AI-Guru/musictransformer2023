"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

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
from source.transformer import TransformerConfig, Transformer
from source.tokenizer import Tokenizer

from dataclasses import dataclass, asdict

@dataclass
class TrainingConfig:

    # Dataset.
    dataset_path = "data/jsfakes4bars/generation"

    # I/O
    out_dir: str = "out"
    eval_interval: int = 200
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # If True, script exits right after the first eval.
    always_save_checkpoint: bool = True  # If True, always save a checkpoint after each eval.
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'.
    
    # wandb logging
    wandb_log: bool = False  # Disabled by default.
    wandb_project: str = "owt"
    wandb_run_name: str = "transformer"  # 'run' + str(time.time()).

    # data
    gradient_accumulation_steps: int = 5 * 8  # Used to simulate larger batch sizes.
    batch_size: int = 128  # If gradient_accumulation_steps > 1, this is the micro-batch size.
    #block_size: int = 384

    # adamw optimizer
    learning_rate: float = 6e-4  # Max learning rate.
    max_iters: int = 600000  # Total number of training iterations.
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # Clip gradients at this value, or disable if == 0.0.

    # learning rate decay settings
    decay_lr: bool = True  # Whether to decay the learning rate.
    warmup_iters: int = 2000  # How many steps to warm up for.
    lr_decay_iters: int = 600000  # Should be ~= max_iters per Chinchilla.
    min_lr: float = 6e-5  # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla.

    # DDP settings
    backend: str = "nccl"  # 'nccl', 'gloo', etc.

    # system
    device: str = "cuda"  # Examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks.
    dtype: str = "bfloat16" if ("torch" in locals() or "torch" in globals()) and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler.
    compile: bool = False  # Use PyTorch 2.0 to compile the model to be faster.

# Get the timestamp as YYYYMMDD-HHMM.
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

# Create the training config.
config = TrainingConfig()

# Set the device.
if config.device == "auto" and torch.cuda.is_available():
    config.device = "cuda"
elif config.device == "auto" and torch.backends.mps.is_available():
    config.device = "mps"
elif config.device == "auto":
    config.device = "cpu"
elif config.device == "cuda" and not torch.cuda.is_available():
    config.device = "cpu"
    print("Warning: device is cuda but cuda is not available. Using cpu instead.")
print(f"config.device: {config.device}")

# Create the model config.
model_config = TransformerConfig()
model_config.vocab_size = 118
model_config.n_layer = 2
model_config.n_head = 2
model_config.n_embd = 128
model_config.dropout = 0.0
model_config.bias = False
model_config.block_size = 384
model_config.bottleneck = "variational" # "simple" or "variational" or "none"
model_config.bottleneck_depth = 2

# Set the output directory.
config.out_dir = os.path.join(config.out_dir, f"transformer_{model_config.bottleneck}_{timestamp}")

# Set the model config.
config.wandb_log = True
config.wandb_project = "bottleneck-transformers"
config.wandb_run_name = f"transformer_{model_config.bottleneck}_{timestamp}"

# Create the output directory.
os.makedirs(config.out_dir, exist_ok=True)

# Save both configs to disk. Map to dict first.
training_config_path = os.path.join(config.out_dir, "training_config.json")
model_config_path = os.path.join(config.out_dir, "model_config.json")
with open(training_config_path, "w") as f:
    json.dump(asdict(config), f, indent=4)
with open(model_config_path, "w") as f:
    json.dump(asdict(model_config), f, indent=4)


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=config.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert config.gradient_accumulation_steps % ddp_world_size == 0
    config.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * model_config.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
#data_dir = os.path.join('data', dataset)
#train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
#val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
#def get_batch(split):
#    data = train_data if split == 'train' else val_data
#    ix = torch.randint(len(data) - block_size, (batch_size,))
#    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
#    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
#    if device_type == 'cuda':
#        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
#        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#    else:
#        x, y = x.to(device), y.to(device)
#    return x, y

# Load the dataset from disk.
# The dataset is already tokenized and split into train and val.
# There are two files in the dataset: train.jsonl and val.jsonl.
assert os.path.exists(config.dataset_path), f"Error: {config.dataset_path} does not exist."
train_dataset_path = os.path.join(config.dataset_path, "train.jsonl")
validate_dataset_path = os.path.join(config.dataset_path, "val.jsonl")
assert os.path.exists(train_dataset_path), f"Error: {train_dataset_path} does not exist."
assert os.path.exists(validate_dataset_path), f"Error: {validate_dataset_path} does not exist."
dataset_train = load_dataset("json", data_files=train_dataset_path, split="train")
dataset_validate = load_dataset("json", data_files=validate_dataset_path, split="train")
print(dataset_train)
print(dataset_validate)

# Create and save the tokenizer.
tokenizer = Tokenizer.from_vocabulary_file(os.path.join(config.dataset_path, "vocabulary.txt"))
tokenizer.save(os.path.join(config.out_dir, "tokenizer.json"))
del tokenizer

def get_batch(split):
    if split == "train":
        dataset = dataset_train
    elif split == "val":
        dataset = dataset_validate
    else:
        raise ValueError(f"Error: split {split} not supported.")
    
    # Get a random batch from the dataset.
    # The dataset is already tokenized.
    # The dataset is already split into train and val.

    # Get random indices.
    indices = np.random.randint(0, len(dataset), config.batch_size).tolist()
    assert len(indices) == config.batch_size, f"Error: len(indices) != batch_size ({len(indices)} != {config.batch_size})."

    # Map to tensors.
    encoder_ids = torch.stack([torch.LongTensor(dataset[i]["encoder_ids"]) for i in indices])
    decoder_ids = torch.stack([torch.LongTensor(dataset[i]["decoder_ids"]) for i in indices])
    target_ids = torch.stack([torch.LongTensor(dataset[i]["target_ids"]) for i in indices])

    # Print the shapes of the tensors and the tensors themselves.
    #print(f"encoder_ids.shape: {encoder_ids.shape}")
    #print(f"decoder_ids.shape: {decoder_ids.shape}")
    #print(f"target_ids.shape: {target_ids.shape}")
    #print(f"encoder_ids: {encoder_ids}")
    #print(f"decoder_ids: {decoder_ids}")
    #print(f"target_ids: {target_ids}")

    # Move the data to the GPU.
    if device_type == "cuda":
        encoder_ids = encoder_ids.pin_memory().to(config.device, non_blocking=True)
        decoder_ids = decoder_ids.pin_memory().to(config.device, non_blocking=True)
        target_ids = target_ids.pin_memory().to(config.device, non_blocking=True)
    else:
        encoder_ids = encoder_ids.to(config.device)
        decoder_ids = decoder_ids.to(config.device)
        target_ids = target_ids.to(config.device)

    return encoder_ids, decoder_ids, target_ids

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
#meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
#if os.path.exists(meta_path):
#    with open(meta_path, 'rb') as f:
#        meta = pickle.load(f)
#    meta_vocab_size = meta['vocab_size']
#    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
#model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
#                  bias=bias, vocab_size=vocab_size, dropout=dropout) # start with model_args from command line
if config.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    #if meta_vocab_size is None:
    #    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    #model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    #model_config = TransformerConfig(**model_args)
    model = Transformer(model_config)
elif config.init_from == 'resume':
    assert False, "Adapt this code to work with the new model."
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
#if block_size < model.config.block_size:
#    model.crop_block_size(block_size)
#    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(config.device)

# Prints the bottleneck shape.
bottleneck_shape = model.get_bottleneck_shape()
print(f"bottleneck shape: {bottleneck_shape} {np.prod(bottleneck_shape):,}")
del bottleneck_shape

# Prit a warning if the device is cpu.
if config.device == "cpu":
    print("Warning: device is cpu. This will be slow.")


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)
if config.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if config.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        reconstruction_losses = torch.zeros(config.eval_iters)
        bottleneck_losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            encoder_ids, decoder_ids, target_ids = get_batch(split)
            with ctx:
                logits, loss, reconstruction_loss, bottleneck_loss = model(encoder_ids, decoder_ids, target_ids)
            losses[k] = loss.item()
            reconstruction_losses[k] = reconstruction_loss.item()
            bottleneck_losses[k] = bottleneck_loss.item()
        out[split] = losses.mean()
        out[f"{split}_reconstruction"] = reconstruction_losses.mean()
        out[f"{split}_bottleneck"] = bottleneck_losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# logging
if config.wandb_log and master_process:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)

# training loop
print("Starting training...")
encoder_ids, decoder_ids, target_ids = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if config.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/reconstruction": losses['train_reconstruction'],
                "val/reconstruction": losses['val_reconstruction'],
                "train/bottleneck": losses['train_bottleneck'],
                "val/bottleneck": losses['val_bottleneck'],
                "lr": lr,
                "mfu": running_mfu * 100, # convert to percentage
            })
        if losses['val'] < best_val_loss or config.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_config': asdict(model_config),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': asdict(config),
                }
                print(f"saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
    if iter_num == 0 and config.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, _, _ = model(encoder_ids, decoder_ids, target_ids)
            loss = loss / config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        encoder_ids, decoder_ids, target_ids = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > config.max_iters:
        break

if ddp:
    destroy_process_group()
