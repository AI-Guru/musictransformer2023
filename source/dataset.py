import os
from dataclasses import dataclass
import datasets
from datasets import load_dataset
import torch

@dataclass
class DatasetConfig:
    dataset_path: str = None
    number_of_processes: int = 1


class Dataset:

    def __init__(self, config: DatasetConfig, device:str, device_type:str):
        self.config = config
        self.device = device
        self.device_type = device_type
        assert os.path.exists(config.dataset_path), f"Error: {config.dataset_path} does not exist."
        self.dataset_path = config.dataset_path
        train_dataset_path = os.path.join(config.dataset_path, "train.jsonl")
        validate_dataset_path = os.path.join(config.dataset_path, "val.jsonl")
        self.dataset_train = load_dataset("json", data_files=train_dataset_path, split="train")
        self.dataset_validate = load_dataset("json", data_files=validate_dataset_path, split="train")
        assert type(self.dataset_train) == datasets.arrow_dataset.Dataset, f"Error: {train_dataset_path} is not a valid dataset. Got {type(self.dataset_train)}."
        assert type(self.dataset_validate) == datasets.arrow_dataset.Dataset, f"Error: {validate_dataset_path} is not a valid dataset. Got {type(self.dataset_validate)}."


    # To string method.
    def __str__(self):
        description = ""
        description += f"Dataset: {self.dataset_path}\n"
        description += f"Train dataset: {self.dataset_train}\n"
        description += f"Validate dataset: {self.dataset_validate}\n"
        return description

    def iterate(self, split:str, shuffle:bool, batch_size:int):
        if split == "train":
            dataset = self.dataset_train
        elif split == "validate":
            dataset = self.dataset_validate
        else:
            raise Exception(f"Error: Unknown split {split}.")
        assert type(dataset) == datasets.arrow_dataset.Dataset, f"Error: {dataset} is not a valid dataset. Got {type(dataset)}."

        # Shuffle if necessary.
        dataset = dataset.shuffle() if shuffle else dataset
        
        # Map.
        def group_batch(batch):
            return {k: [v] for k, v in batch.items()}
        dataset = dataset.map(group_batch, batched=True, batch_size=batch_size, num_proc=self.config.number_of_processes)
        
        for batch in dataset:
            encoder_ids = batch["encoder_ids"]
            decoder_ids = batch["decoder_ids"]
            target_ids = batch["target_ids"]

            # Convert to torch tensors.
            encoder_ids = torch.tensor(encoder_ids).long()
            decoder_ids = torch.tensor(decoder_ids).long()
            target_ids = torch.tensor(target_ids).long()

            # Move the data to the GPU.
            if self.device_type == "cuda":
                encoder_ids = encoder_ids.pin_memory().to(self.device, non_blocking=True)
                decoder_ids = decoder_ids.pin_memory().to(self.device, non_blocking=True)
                target_ids = target_ids.pin_memory().to(self.device, non_blocking=True)
            else:
                encoder_ids = encoder_ids.to(self.device)
                decoder_ids = decoder_ids.to(self.device)
                target_ids = target_ids.to(self.device)
            
            # Yield the data.
            yield encoder_ids, decoder_ids, target_ids
