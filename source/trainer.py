import os
import time
import datetime
import math
import pickle
from contextlib import nullcontext
import json
import multiprocessing
import glob

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.append(".")

from source.tokenizer import Tokenizer

from dataclasses import dataclass, asdict
from typing import Union

@dataclass
class TrainerConfig:

    # General.
    num_epochs: int = 500

    # Dataset.
    use_padding_mask: bool = False

    # I/O
    out_dir: str = "out"
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'.
    
    # When to evaluate.
    eval_every: int = 1
    eval_mode: str = "epochs" # "epochs" or "steps"

    # When to log.
    log_every: int = 1
    log_mode: str = "epochs"

    # When to save.
    save_every: int = 50
    save_mode: str = "epochs"
    save_best: bool = True
    save_last: bool = True

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
    max_iters: int = 15_000  # Total number of training iterations.
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # Clip gradients at this value, or disable if == 0.0.

    # Learning rate decay settings.
    decay_lr: bool = True  # Whether to decay the learning rate.
    warmup_iters: int = 2000  # How many steps to warm up for.
    lr_decay_iters: int = 15_000  # Should be ~= max_iters per Chinchilla.
    min_lr: float = 6e-5  # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla.

    # Bottleneck loss settings.
    bottleneck_loss_coefficient: float = 0.0  # How much to weight the bottleneck loss.
    bottleneck_loss_coefficient_max: float = 1.0  # The maximum bottleneck loss coefficient.
    bottleneck_loss_iterations: int = 15_000  # How many iterations to ramp up the bottleneck loss coefficient.

    # DDP settings
    backend: str = "nccl"  # 'nccl', 'gloo', etc.

    # Debugging.
    find_not_updated_layers:bool = False
    stop_on_vanishing_gradient: bool = False
    max_eval_steps: int = 0  # If > 0, only evaluate this many steps.
    log_grad_norm: bool = False

    # system
    device: str = "auto"  # Examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks.
    dtype: str = "bfloat16" if ("torch" in locals() or "torch" in globals()) and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler.
    compile: bool = True  # Use PyTorch 2.0 to compile the model to be faster.

    def __post_init__(self):

        self.validate()

    def validate(self):
        # General.
        assert isinstance(self.num_epochs, int) and self.num_epochs > 0, "num_epochs must be a positive integer"

        # Dataset.
        assert isinstance(self.use_padding_mask, bool), "use_padding_mask must be a boolean"

        # I/O
        assert isinstance(self.out_dir, str), "out_dir must be a string"
        assert self.init_from in ['scratch', 'resume', 'gpt2*'], "init_from must be 'scratch', 'resume', or 'gpt2*'"

        # When to evaluate.
        assert isinstance(self.eval_every, int) and self.eval_every > 0, "eval_every must be a positive integer"
        assert self.eval_mode in ['epochs', 'steps'], "eval_mode must be 'epochs' or 'steps'"

        # When to log.
        assert isinstance(self.log_every, int) and self.log_every > 0, "log_every must be a positive integer"
        assert self.log_mode in ['epochs', 'steps'], "log_mode must be 'epochs' or 'steps'"

        # When to save.
        assert isinstance(self.save_every, int) and self.save_every > 0, "save_every must be a positive integer"
        assert self.save_mode in ['epochs', 'steps'], "save_mode must be 'epochs' or 'steps'"
        assert isinstance(self.save_best, bool), "save_best must be a boolean"
        assert isinstance(self.save_last, bool), "save_last must be a boolean"

        # W&B logging.
        assert isinstance(self.wandb_log, bool), "wandb_log must be a boolean"
        assert isinstance(self.wandb_project, str), "wandb_project must be a string"
        assert isinstance(self.wandb_run_name, str), "wandb_run_name must be a string"

        # data
        assert isinstance(self.gradient_accumulation_steps, int) and self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be a positive integer"
        assert isinstance(self.batch_size, int) and self.batch_size > 0, "batch_size must be a positive integer"

        # Optimizer settings.
        assert isinstance(self.learning_rate, float) and self.learning_rate > 0, "learning_rate must be a positive float"
        assert isinstance(self.max_iters, int) and self.max_iters > 0, "max_iters must be a positive integer"
        assert isinstance(self.weight_decay, float) and self.weight_decay >= 0, "weight_decay must be a non-negative float"
        assert isinstance(self.beta1, float) and 0 <= self.beta1 <= 1, "beta1 must be a float between 0 and 1"
        assert isinstance(self.beta2, float) and 0 <= self.beta2 <= 1, "beta2 must be a float between 0 and 1"
        assert isinstance(self.grad_clip, float) and self.grad_clip >= 0, "grad_clip must be a non-negative float"

        # learning rate decay settings
        assert isinstance(self.decay_lr, bool), "decay_lr must be a boolean"
        assert isinstance(self.warmup_iters, int) and self.warmup_iters >= 0, "warmup_iters must be a non-negative integer"
        assert isinstance(self.lr_decay_iters, int) and self.lr_decay_iters > 0, "lr_decay_iters must be a positive integer"
        assert isinstance(self.min_lr, float) and self.min_lr > 0, "min_lr must be a positive float"

        # DDP settings
        assert isinstance(self.backend, str), "backend must be a string"

        # system
        assert isinstance(self.device, str), "device must be a string"
        assert self.dtype in ["float32", "bfloat16", "float16"], "dtype must be 'float32', 'bfloat16', or 'float16'"
        assert isinstance(self.compile, bool), "compile must be a boolean"

        # Debugging.
        assert isinstance(self.find_not_updated_layers, bool), "find_not_updated_layers must be a boolean"
        assert isinstance(self.max_eval_steps, int) and self.max_eval_steps >= 0, "max_eval_steps must be a non-negative integer"
        assert isinstance(self.log_grad_norm, bool), "log_grad_norm must be a boolean"


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
        config.validate()
        self.config = config

        self.device_type = "cuda" if "cuda" in config.device else "cpu"

    def train(self, model, dataset):

        # Make sure the output folder exists.
        os.makedirs(self.config.out_dir, exist_ok=True)

        # Send the model to the device.
        print(f"Sending the model to {self.config.device}...")
        model = model.to(self.config.device)

        # Test if the model can be saved and loaded.
        print("Testing if the model can be saved and loaded...")
        checkpoint_path = os.path.join(self.config.out_dir, "test_checkpoint.pt")
        checkpoint_path = self.save_checkpoint(
            model=model,
            optimizer=None,
            step=None,
            epoch=None,
            checkpoint_name="test_checkpoint.pt",
        )
        # Get the class of the model.
        model_class = model.__class__
        _ = model_class.load(checkpoint_path)
        os.remove(checkpoint_path)
        print("Model can be saved and loaded.")

        # Inform the dataset about the device.
        dataset.set_device(self.config.device, self.device_type)

        # Save the configs of the model and the trainer and the dataset.
        print("Saving the configs...")
        model_config_path = os.path.join(self.config.out_dir, "model_config.json")
        with open(model_config_path, "w") as model_config_file:
            json.dump(asdict(model.config), model_config_file, indent=4)
        trainer_config_path = os.path.join(self.config.out_dir, "trainer_config.json")
        with open(trainer_config_path, "w") as trainer_config_file:
            json.dump(asdict(self.config), trainer_config_file, indent=4)
        dataset_config_path = os.path.join(self.config.out_dir, "dataset_config.json")
        with open(dataset_config_path, "w") as dataset_config_file:
            json.dump(asdict(dataset.config), dataset_config_file, indent=4)

        # Create and save the tokenizer.
        print("Creating and saving the tokenizer...")
        tokenizer_vocabulary_path = os.path.join(self.config.out_dir, "tokenizer.json")
        assert os.path.exists(os.path.join(dataset.config.dataset_path, "vocabulary.txt")), f"Error: vocabulary.txt does not exist in {config.dataset_path}."
        tokenizer = Tokenizer.from_vocabulary_file(os.path.join(dataset.config.dataset_path, "vocabulary.txt"))
        tokenizer.save(tokenizer_vocabulary_path)
        del tokenizer

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

        # learning rate decay scheduler (cosine with warmup)
        def get_learning_rate(iteration):
            # 1) linear warmup for warmup_iters steps
            if iteration < self.config.warmup_iters:
                return self.config.learning_rate * iteration / self.config.warmup_iters
            # 2) if iteration > lr_decay_iters, return min learning rate
            if iteration > self.config.lr_decay_iters:
                return self.config.min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (iteration - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

        # The bottleneck loss coefficient scheduler.
        def get_bottleneck_loss_coefficient(iteration):
            if self.config.bottleneck_loss_coefficient == self.config.bottleneck_loss_coefficient_max:
                return self.config.bottleneck_loss_coefficient_max
            factor = min(1.0, iteration / self.config.bottleneck_loss_iterations)
            return self.config.bottleneck_loss_coefficient * (1.0 - factor) + self.config.bottleneck_loss_coefficient_max * factor

        # logging
        if self.config.wandb_log:# and master_process: #TODO
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "training_config": asdict(self.config),
                    "model_config": asdict(model.config),
                    "dataset_config": asdict(dataset.config),
                }
            )

        def create_losses_dict():
            losses_dict = {}
            for split in ["train", "val"]:
                for loss_type in ["loss", "reconstruction", "bottleneck"]:
                    losses_dict[f"{split}/{loss_type}"] = []
            return losses_dict
        
        def create_kpi_dict():
            kpi_dict = {}
            if self.config.log_grad_norm:
                kpi_types = ["grad_norm/total"]
                for module_name, _ in model.get_modules().items():
                    kpi_types.append(f"grad_norm/{module_name}")
                for kpi_type in kpi_types:
                    kpi_dict[kpi_type] = []
            return kpi_dict
        
        # Go through the epochs.
        print("Starting training...")
        total_training_steps = 0
        training_start_time = time.time()
        model.train()
        epoch_data_iterator = None
        current_epoch = 1
        stop_training = False
        losses_dict = create_losses_dict()
        best_losses = { key: float ("inf") for key in losses_dict.keys() }
        kpi_dict = create_kpi_dict()
        vanished_gradient = False
        while True:

            # Start the next epoch.
            if epoch_data_iterator is None:
                print(f"Starting epoch {current_epoch}/{self.config.num_epochs}...")
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
                print(f"Epoch {current_epoch}/{self.config.num_epochs} elapsed time: {datetime.timedelta(seconds=epoch_elapsed_time)}")

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

                # Unpack the batch.
                encoder_ids_train, decoder_ids_train, target_ids_train, padding_masks_train = batch_train

                # Padding mask.
                if not self.config.use_padding_mask:
                    padding_masks_train = None

                # Get the learning rate.
                lr = get_learning_rate(total_training_steps) if self.config.decay_lr else self.config.learning_rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Get the bottleneck loss coefficient.
                bottleneck_loss_coefficient = get_bottleneck_loss_coefficient(total_training_steps)

                # Print.
                print(f"Epoch={current_epoch:,}/{self.config.num_epochs} step={total_training_steps:,} lr={lr:.6f} blc={bottleneck_loss_coefficient:.6f} vanished: {vanished_gradient}", end="\r")

                # Forward pass and get the loss.
                if model.family == "encoderdecoder":
                    _, reconstruction_loss, bottleneck_loss = model(
                        encoder_ids_train,
                        decoder_ids_train,
                        target_ids=target_ids_train,
                        padding_mask=padding_masks_train,
                    )
                elif model.family == "encoder":
                    _, reconstruction_loss, bottleneck_loss = model(
                        encoder_ids_train,
                        target_ids=encoder_ids_train,
                        padding_mask=padding_masks_train,
                    )
                else:
                    raise Exception(f"Unknown model family: {model.family}")
                
                # Add the bottleneck loss if it is not None.
                if bottleneck_loss is not None:
                    loss = reconstruction_loss + bottleneck_loss_coefficient * bottleneck_loss
                    losses_dict["train/bottleneck"].append(bottleneck_loss.item())
                else:
                    loss = reconstruction_loss
                losses_dict["train/loss"].append(loss.item())
                losses_dict["train/reconstruction"].append(reconstruction_loss.item())

                # Backward pass.
                assert isinstance(loss, torch.Tensor), "Error: loss is not a torch.Tensor."
                scaler.scale(loss).backward()
                
                # Clip the gradient.
                if self.config.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                # Step the optimizer and scaler if training in fp16.
                scaler.step(optimizer)
                scaler.update()

                # Find all the layers of the model that were not updated.
                # Output will happen only once.
                if (self.config.find_not_updated_layers or self.config.stop_on_vanishing_gradient) and not vanished_gradient:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                print(f"Warning: {name} has no gradient.")
                                vanished_gradient = True
                            elif torch.all(param.grad == 0.0):
                                print(f"Warning: {name} has a gradient of all zeros.")
                                vanished_gradient = True
                            elif torch.any(torch.isnan(param.grad)):
                                print(f"Warning: {name} has a gradient of all NaNs.")
                                vanished_gradient = True
                    if vanished_gradient and self.config.stop_on_vanishing_gradient:
                        print("Stopping script because of vanished gradient.")
                        exit(0)

                # Log the gradient norm. Do this per module.
                if self.config.log_grad_norm:
                    for module_name, module in model.get_modules().items():
                        for _, parameter in module.named_parameters():
                            if parameter.requires_grad:
                                key = f"grad_norm/{module_name}"
                                if parameter.grad is not None:
                                    norm = torch.linalg.norm(parameter.grad).item()
                                    kpi_dict["grad_norm/total"].append(norm)
                                    kpi_dict[key].append(norm)
                                else:
                                    kpi_dict["grad_norm/total"].append(0.0)
                                    kpi_dict[key].append(0.0)

                # Flush the gradients as soon as we can, no need for this memory anymore.
                optimizer.zero_grad(set_to_none=True)

                # Delete other variables to free up memory.
                del encoder_ids_train
                del decoder_ids_train
                del target_ids_train
                del loss
                del reconstruction_loss
                del bottleneck_loss

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

                # Let us try emptying the cache here.
                torch.cuda.empty_cache()

                print(f"Validate epoch {current_epoch:,} step {total_training_steps:,}...")
                model.eval()
                validation_steps = 0
                for encoder_ids_validate, decoder_ids_validate, target_ids_validate, padding_masks_validate in dataset.iterate(split="validate", shuffle=False, batch_size=self.config.batch_size):
                    print(f"Validation step {validation_steps:,}", end="\r")
                    validation_steps += 1
                    
                    # Stop evaluation here.
                    if self.config.max_eval_steps > 0 and validation_steps >= self.config.max_eval_steps:
                        break
                    
                    # Padding mask.
                    if not self.config.use_padding_mask:
                        padding_masks_validate = None

                    # Forward pass and get the losses.
                    if model.family == "encoderdecoder":
                        _, reconstruction_loss, bottleneck_loss = model(
                            encoder_ids=encoder_ids_validate, 
                            decoder_ids=decoder_ids_validate,
                            target_ids=target_ids_validate,
                            padding_mask=padding_masks_validate,    
                        )
                    elif model.family == "encoder":
                        _, reconstruction_loss, bottleneck_loss = model(
                            encoder_ids=encoder_ids_validate, 
                            target_ids=encoder_ids_validate,
                            padding_mask=padding_masks_validate,    
                        )
                    else:
                        raise Exception(f"Unknown model family: {model.family}")

                    # Update epoch loss.
                    if bottleneck_loss is not None:
                        loss = reconstruction_loss + bottleneck_loss_coefficient * bottleneck_loss
                        losses_dict["val/bottleneck"].append(bottleneck_loss.item())
                    else:
                        loss = reconstruction_loss
                    losses_dict["val/loss"].append(loss.item())
                    losses_dict["val/reconstruction"].append(reconstruction_loss.item())
                    
                print("")
                torch.cuda.empty_cache()
                model.train()

                # Save the best model.
                if self.config.save_best:

                    for key, value in losses_dict.items():
                        if np.mean(value) < best_losses[key]:
                            best_losses[key] = np.mean(value)
                            print(f"New best {key}: {best_losses[key]:.4f}")

                            # Delete all checkpoints that have _best in the name.
                            for checkpoint_path in glob.glob(os.path.join(self.config.out_dir, f"*_{key.replace('/', '-')}_best.pt")):
                                os.remove(checkpoint_path)

                            # Save the best model.                    
                            print(f"Saving best model...")
                            checkpoint_name = f"checkpoint_{current_epoch:04d}_{total_training_steps:08d}_{key.replace('/', '-')}_best.pt"
                            self.save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                step=total_training_steps,
                                epoch=current_epoch,
                                checkpoint_name=checkpoint_name,
                            )

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
                log_dict["bottleneck_loss_coefficient"] = bottleneck_loss_coefficient
                #log_dict["mfu"] = mfu

                for key, values in kpi_dict.items():
                    if key.startswith("grad_norm"):
                        log_dict[f"{key}_mean"] = np.mean(values)
                    else:
                        raise Exception(f"Unknown key: {key}")
                kpi_dict = create_kpi_dict()

                # Log to terminal.
                print(f"Log at epoch {current_epoch}/{self.config.num_epochs} step {total_training_steps}...")
                #print(" ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()]))
                
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
                checkpoint_name = f"checkpoint_{current_epoch:04d}_{total_training_steps:08d}.pt"
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=total_training_steps,
                    epoch=current_epoch,
                    checkpoint_name=checkpoint_name,
                )

            # Stop the training if we are done.
            if stop_training:
                break

        # Save the final model.
        if self.config.save_last:
            print("Saving final model...")
            checkpoint_name = f"checkpoint_{current_epoch:04d}_{total_training_steps:08d}_last.pt"
            self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=total_training_steps,
                epoch=current_epoch,
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
        
        # Create the checkpoint dictionary.
        checkpoint = {}
        if model is not None:
            checkpoint["model"] = model.state_dict()
            checkpoint["model_config"] = asdict(model.config)
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if step is not None:
            checkpoint["step"] = step
        if epoch is not None:
            checkpoint["epoch"] = epoch
        #checkpoint["config"] = asdict(self.config)
        
        # Save the checkpoint.
        checkpoint_path = os.path.join(self.config.out_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

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
