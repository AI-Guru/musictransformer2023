import os
from dataclasses import dataclass
from typing import Union
import datasets
from datasets import load_dataset
import torch
import numpy as np

@dataclass
class DatasetConfig:
    dataset_path: str = None
    number_of_processes: int = 1

    # Token dropout settings.
    # Note: Copied verbatim from trainer.
    token_dropout: bool = True  # Whether to use token dropout.
    token_dropout_probability: float = 0.2
    token_dropout_encoder: bool = False  # Whether to use token dropout in the encoder.
    token_dropout_training: bool = True  # Whether to only use token dropout during training.
    token_dropout_validation: bool = False  # Whether to only use token dropout during validation.
    token_dropout_mask_token: Union[int, str] = "all"  # Which token to use for masking. 'all' or an integer.

    def __post_init__(self):
        self.validate()

    def validate(self):
        assert self.dataset_path is not None, "Error: Dataset path not set."
        assert os.path.exists(self.dataset_path), f"Error: {self.dataset_path} does not exist."
        assert self.number_of_processes > 0, f"Error: Number of processes must be greater than 0. Got {self.number_of_processes}."

         # Token dropout settings.
        assert isinstance(self.token_dropout, bool), "token_dropout must be a boolean"
        assert isinstance(self.token_dropout_probability, float) and 0 <= self.token_dropout_probability <= 1, "token_dropout_probability must be a float between 0 and 1"
        assert isinstance(self.token_dropout_encoder, bool), "token_dropout_encoder must be a boolean"
        assert isinstance(self.token_dropout_training, bool), "token_dropout_training must be a boolean"
        assert isinstance(self.token_dropout_validation, bool), "token_dropout_validation must be a boolean"
        assert isinstance(self.token_dropout_mask_token, (int, str)), "token_dropout_mask_token must be an int or string"


class Dataset:

    def __init__(self, config: DatasetConfig):
        
        # Validate the config.
        config.validate()

        # Save the config.
        self.config = config
        
        # Load the datasets.
        assert os.path.exists(config.dataset_path), f"Error: {config.dataset_path} does not exist."
        self.dataset_path = config.dataset_path
        train_dataset_path = os.path.join(config.dataset_path, "train.jsonl")
        validate_dataset_path = os.path.join(config.dataset_path, "val.jsonl")
        self.dataset_train = load_dataset("json", data_files=train_dataset_path, split="train")
        self.dataset_validate = load_dataset("json", data_files=validate_dataset_path, split="train")
        assert type(self.dataset_train) == datasets.arrow_dataset.Dataset, f"Error: {train_dataset_path} is not a valid dataset. Got {type(self.dataset_train)}."
        assert type(self.dataset_validate) == datasets.arrow_dataset.Dataset, f"Error: {validate_dataset_path} is not a valid dataset. Got {type(self.dataset_validate)}."
        
        # Load the vocabulary.
        vocabulary_path = os.path.join(config.dataset_path, "vocabulary.txt")
        assert os.path.exists(vocabulary_path), f"Error: {vocabulary_path} does not exist."
        self.vocabulary = open(vocabulary_path).read().split("\n")
        
        # Set the device. Need to be able to set this after the fact.
        self.device = None
        self.device_type = None

    def set_device(self, device:str, device_type:str):
        self.device = device
        self.device_type = device_type

    # To string method.
    def __str__(self):
        description = ""
        description += f"Dataset: {self.dataset_path}\n"
        description += f"Train dataset: {self.dataset_train}\n"
        description += f"Validate dataset: {self.dataset_validate}\n"
        return description

    def iterate(self, split:str, shuffle:bool, batch_size:int):
        
        # Get the dataset.
        if split == "train":
            dataset = self.dataset_train
        elif split == "validate":
            dataset = self.dataset_validate
        else:
            raise Exception(f"Error: Unknown split {split}.")
        assert type(dataset) == datasets.arrow_dataset.Dataset, f"Error: {dataset} is not a valid dataset. Got {type(dataset)}."

        # Shuffle if necessary.
        dataset = dataset.shuffle() if shuffle else dataset

        # Determine if we should do token dropout.
        do_token_dropout = False
        if self.config.token_dropout:
            if (split == "train" and self.config.token_dropout_training) or (split == "validate" and self.config.token_dropout_validation):
                do_token_dropout = True
        
        # If so get the mask token and the other relevant variables.
        if do_token_dropout:
            if self.config.token_dropout_mask_token == "all":
                mask_tokens = list(range(len(self.vocabulary)))
            elif isinstance(self.config.token_dropout_mask_token, str):
                assert self.config.token_dropout_mask_token in self.vocabulary, f"Error: {self.config.token_dropout_mask_token} not in vocabulary."
                mask_tokens = [self.vocabulary.index(self.config.token_dropout_mask_token)]
            else:
                mask_tokens = [self.config.token_dropout_mask_token]
            token_dropout_probability = self.config.token_dropout_probability
            do_token_dropout_decoder = True
            do_token_dropout_encoder = self.config.token_dropout_encoder

        # Define the token dropout function.
        def apply_token_dropout(ids_batch):
            ids_batch_augmented = []
            for ids in ids_batch:
                ids_augmented = []
                for id in ids:
                    if np.random.rand() < token_dropout_probability:
                        ids_augmented.append(np.random.choice(mask_tokens))
                    else:
                        ids_augmented.append(id)
                ids_batch_augmented.append(ids_augmented)
            return ids_batch_augmented

        # Map into batches.
        def group_batch(batch):
            return {k: [v] for k, v in batch.items()}
        dataset = dataset.map(group_batch, batched=True, batch_size=batch_size, num_proc=self.config.number_of_processes)

        # Yield each batch.
        for batch in dataset:
            encoder_ids = batch["encoder_ids"]
            decoder_ids = batch["decoder_ids"]
            target_ids = batch["target_ids"]

            # Do token dropout if necessary.
            if do_token_dropout_encoder:
                encoder_ids = apply_token_dropout(encoder_ids)
            if do_token_dropout_decoder:
                decoder_ids = apply_token_dropout(decoder_ids)

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
