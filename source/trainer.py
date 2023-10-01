import os
import time
import datetime
import math
import pickle
from contextlib import nullcontext
import json
import multiprocessing

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from datasets import load_dataset # huggingface datasets

import sys
sys.path.append(".")

#from model import GPTConfig, GPT
from source.dataset import DatasetConfig, Dataset
#from source.transformer import TransformerConfig, Transformer
from source.tokenizer import Tokenizer

from dataclasses import dataclass, asdict

@dataclass
class TrainerConfig:

    # General.
    num_epochs: int = 100

    # Dataset.
    dataset_path = "data/jsfakes4bars/generation"

    # I/O
    out_dir: str = "out"
    #eval_interval: int = 200
    #log_interval: int = 1
    #eval_iters: int = 200
    #eval_only: bool = False  # If True, script exits right after the first eval.
    #always_save_checkpoint: bool = True  # If True, always save a checkpoint after each eval.
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'.
    
    # When to evaluate.
    eval_every: int = 1
    eval_mode: str = "epochs" # "epochs" or "steps"

    # When to log.
    log_every: int = 1
    log_mode: str = "epochs"

    # When to save.
    save_every: int = 1
    save_mode: str = "epochs"

    # W&B logging.
    wandb_log: bool = False  # Disabled by default.
    wandb_project: str = "owt"
    wandb_run_name: str = "transformer"  # 'run' + str(time.time()).

    # data
    gradient_accumulation_steps: int = 5 * 8  # Used to simulate larger batch sizes.
    batch_size: int = 128  # If gradient_accumulation_steps > 1, this is the micro-batch size.
    #block_size: int = 384

    # Optimizer settings.
    learning_rate: float = 6e-4  # Max learning rate.
    max_iters: int = 10_000  # Total number of training iterations.
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # Clip gradients at this value, or disable if == 0.0.

    # learning rate decay settings
    decay_lr: bool = True  # Whether to decay the learning rate.
    warmup_iters: int = 2000  # How many steps to warm up for.
    lr_decay_iters: int = 10_000  # Should be ~= max_iters per Chinchilla.
    min_lr: float = 6e-5  # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla.

    # DDP settings
    backend: str = "nccl"  # 'nccl', 'gloo', etc.

    # system
    device: str = "auto"  # Examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks.
    dtype: str = "bfloat16" if ("torch" in locals() or "torch" in globals()) and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler.
    compile: bool = True  # Use PyTorch 2.0 to compile the model to be faster.


class Trainer:

    def __init__(self, config:TrainerConfig):

        # Set the device.
        if config.device == "auto" and torch.cuda.is_available():
            config.device = "cuda"
        #elif config.device == "auto" and torch.backends.mps.is_available():
        #    config.device = "mps"
        elif config.device == "auto":
            config.device = "cpu"
        elif config.device == "cuda" and not torch.cuda.is_available():
            config.device = "cpu"
            print("Warning: device is cuda but cuda is not available. Using cpu instead.")
        if config.device == "cpu":
            print("Warning: device is cpu. This will be slow.")
        else:
            print(f"Device: {config.device}")
        self.config = config

        self.device_type = "cuda" if "cuda" in config.device else "cpu"


    def train(self, model):

        # Make sure the output folder exists.
        os.makedirs(self.config.out_dir, exist_ok=True)

        # Send the model to the device.
        print(f"Sending the model to {self.config.device}...")
        model = model.to(self.config.device)

        # Create the dataset.
        print("Creating dataset...")
        dataset_config = DatasetConfig()
        dataset_config.dataset_path = self.config.dataset_path
        dataset_config.number_of_processes = multiprocessing.cpu_count()
        dataset_config.dataset_path = "data/jsfakes4bars/generation"
        dataset = Dataset(dataset_config, device=self.config.device, device_type=self.device_type)
        print(dataset)

        #torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.dtype]
        ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)


        # Initialize a GradScaler. If enabled=False scaler is a no-op
        print("Initializing GradScaler...")
        scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == 'float16'))

        # Initialize the optimizer.
        print("Initializing optimizer...")
        optimizer = model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, (self.config.beta1, self.config.beta2), self.device_type)
        if self.config.init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None # free up memory
        
        # Compile the model.
        if self.config.compile:
            # Check if the pytorch version is 2.0 or higher.
            if not hasattr(torch, "compile"):
                print("PyTorch 2.0 or higher is required to use the compile option.")
            else:
                print("Compiling the model...")
                unoptimized_model = model
                model = torch.compile(model)

        # Wrap model into DDP container.
        #if ddp:
        #    assert False, "Error: DDP is not implemented yet."
        #    model = DDP(model, device_ids=[ddp_local_rank])

        # helps estimate an arbitrarily accurate loss over either split using many batches
        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(self.config.eval_iters)
                reconstruction_losses = torch.zeros(self.config.eval_iters)
                bottleneck_losses = torch.zeros(self.config.eval_iters)
                for k in range(self.config.eval_iters):
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
            if it < self.config.warmup_iters:
                return self.config.learning_rate * it / self.config.warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > self.config.lr_decay_iters:
                return self.config.min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

        # logging
        if self.config.wandb_log:# and master_process: #TODO
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "training_config": asdict(self.config),
                    "model_config": asdict(model.config),
                }
            )

        def create_losses_dict():
            losses_dict = {}
            for split in ["train", "val"]:
                for loss_type in ["loss", "reconstruction", "bottleneck"]:
                    losses_dict[f"{split}/{loss_type}"] = []
            return losses_dict
        
        # Go through the epochs.
        print("Starting training...")
        total_training_steps = 0
        training_start_time = time.time()
        model.train()
        epoch_data_iterator = None
        current_epoch = 0
        stop_training = False
        losses_dict = create_losses_dict()
        while True:

            # Start the next epoch.
            if epoch_data_iterator is None:
                print(f"Starting epoch {current_epoch}...")
                epoch_start_time = time.time()
                epoch_training_steps = 0
                epoch_data_iterator = dataset.iterate(split="train", shuffle=True, batch_size=self.config.batch_size)
                epoch_done = False

            # Get the next batch.
            try:
                batch_train = next(epoch_data_iterator)
            
            # End of epoch.
            except StopIteration:
                epoch_elapsed_time = time.time() - epoch_start_time
                print(f"Epoch {current_epoch} elapsed time: {datetime.timedelta(seconds=epoch_elapsed_time)}")

                epoch_data_iterator = None
                current_epoch += 1
                epoch_done = True

                # Quit the training if we are done. Leave the loop.
                if current_epoch >= self.config.num_epochs:
                    stop_training = True

            # Do the training step.
            do_step = False
            if not epoch_done and not stop_training:
                do_step = True
            #do_step = False # TODO Remove this.
            if do_step:

                print(f"Epoch {current_epoch} step {total_training_steps}", end="\r")

                # Unpack the batch.
                encoder_ids_train, decoder_ids_train, target_ids_train = batch_train

                # Get the learning rate.
                lr = get_lr(total_training_steps) if self.config.decay_lr else self.config.learning_rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Forward pass and get the loss.
                _, loss, reconstruction_loss, bottleneck_loss = model(encoder_ids_train, decoder_ids_train, target_ids_train)
                
                # Update epoch loss.
                losses_dict["train/loss"].append(loss.item())
                losses_dict["train/reconstruction"].append(reconstruction_loss.item())
                losses_dict["train/bottleneck"].append(bottleneck_loss.item())

                # Backward pass.
                scaler.scale(loss).backward()
                
                # Clip the gradient.
                if self.config.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                # Step the optimizer and scaler if training in fp16.
                scaler.step(optimizer)
                scaler.update()
                
                # Flush the gradients as soon as we can, no need for this memory anymore.
                optimizer.zero_grad(set_to_none=True)

            # Increment the total training steps.
            total_training_steps += 1
            epoch_training_steps += 1
            #print(f"Total training steps: {total_training_steps} epoch training steps: {epoch_training_steps} epoch: {current_epoch}")

            # Do the validation.
            do_validate = False
            if self.config.eval_mode == "epochs" and epoch_done and current_epoch % self.config.eval_every == 0:
                do_validate = True
            elif self.config.eval_mode == "steps" and total_training_steps % self.config.eval_every == 0:
                do_validate = True
            if do_validate:
                print(f"Validate at epoch {current_epoch} step {total_training_steps}...")
                model.eval()
                for encoder_ids_validate, decoder_ids_validate, target_ids_validate in dataset.iterate(split="validate", shuffle=False, batch_size=self.config.batch_size):
                    # Forward pass and get the losses.
                    _, loss, reconstruction_loss, bottleneck_loss = model(encoder_ids_validate, decoder_ids_validate, target_ids_validate)

                    # Update epoch loss.
                    losses_dict["val/loss"].append(loss.item())
                    losses_dict["val/reconstruction"].append(reconstruction_loss.item())
                    losses_dict["val/bottleneck"].append(bottleneck_loss.item())
                model.train()

            # Log everything.
            do_log = False
            if self.config.log_mode == "epochs" and epoch_done and current_epoch % self.config.log_every == 0:
                do_log = True
            elif self.config.log_mode == "steps" and total_training_steps % self.config.log_every == 0:
                do_log = True
            if do_log:
                losses_dict_mean = {k: np.mean(v) for k, v in losses_dict.items()}
                losses_dict = create_losses_dict() 
                
                log_dict = { k: v for k, v in losses_dict_mean.items() }
                log_dict["step"] = total_training_steps
                log_dict["lr"] = lr
                log_dict["epoch"] = current_epoch
                #log_dict["mfu"] = mfu

                # Log to terminal.
                print(f"Log at epoch {current_epoch} step {total_training_steps}...")
                print(" ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()]))
                
                # Log to wandb.
                if self.config.wandb_log:
                    wandb.log(log_dict, step=total_training_steps)

            # Save the checkpoint.
            do_save_checkpoint = False
            if self.config.save_mode == "epochs" and epoch_done and current_epoch % self.config.save_every == 0:
                do_save_checkpoint = True
            elif self.config.save_mode == "steps" and total_training_steps % self.config.save_every == 0:
                do_save_checkpoint = True
            if do_save_checkpoint:
                print(f"Save checkpoint at epoch {current_epoch} step {total_training_steps}...")
                checkpoint_name = f"checkpoint_{current_epoch:04d}{total_training_steps:08d}.pt"
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=total_training_steps,
                    epoch=current_epoch
                    checkpoint_name=checkpoint_name,
                )

            # Stop the training if we are done.
            if stop_training:
                break

        # Save the final model.
        print("Saving final model...")
        checkpoint_name = f"checkpoint_{current_epoch:04d}{total_training_steps:08d}_last.pt"
        self.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=total_training_steps,
            epoch=current_epoch
            checkpoint_name=checkpoint_name,
        )

        # Clean up.
        if self.config.wandb_log:
            wandb.finish()
        torch.cuda.empty_cache()

        # Done.
        training_elapsed_time = time.time() - training_start_time
        print(f"Total steps: {total_training_steps} elapsed time: {datetime.timedelta(seconds=training_elapsed_time)}")

    def save_checkpoint(self, model, optimizer, step, epoch, checkpoint_name):
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model.config),
            "step": total_training_steps,
            "epoch": current_epoch,
            #"best_val_loss": best_val_loss,
            "config": asdict(self.config),
        }
        checkpoint_path = os.path.join(self.config.out_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def info(self, model):

        # Print model config.
        print("Model config:")
        for key, value in model.config.__dict__.items():
            print(f"  {key}: {value}")

        # Print the latent space size.
        shape = model.get_bottleneck_shape()
        print(f"Latent space size: {shape} ({np.prod(shape)} units)")

        # Done.
        print("")
