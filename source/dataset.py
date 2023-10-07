import os
from dataclasses import dataclass
from typing import Union
import datasets
from datasets import load_dataset, load_from_disk
import torch
import numpy as np
from torch.utils.data import DataLoader


@dataclass
class DatasetConfig:
    dataset_path: str = None
    number_of_processes: Union[str, int] = "auto"

    # Token dropout settings.
    token_dropout: bool = False  # Whether to use token dropout.
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
        assert self.number_of_processes == "auto" or self.number_of_processes > 0, "Error: number_of_processes must be 'auto' or a positive integer."

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

        # Set the number of processes.
        if config.number_of_processes == "auto":
            config.number_of_processes = os.cpu_count()

        # Save the config.
        self.config = config
        
        # Load the datasets.
        assert os.path.exists(config.dataset_path), f"Error: {config.dataset_path} does not exist."
        self.dataset_path = config.dataset_path

        # The dataset exists as a HF dataset on the HD.
        print(f"Load dataset {config.dataset_path}...")
        dataset = load_from_disk(config.dataset_path)
        dataset_train = dataset["train"]
        dataset_validate = dataset["val"]
        assert type(dataset_train) == datasets.arrow_dataset.Dataset, f"Error: {dataset_train} is not a valid dataset. Got {type(dataset_train)}."
        assert type(dataset_validate) == datasets.arrow_dataset.Dataset, f"Error: {dataset_validate} is not a valid dataset. Got {type(dataset_validate)}."
        self.dataset_train = dataset_train
        self.dataset_validate = dataset_validate
        
        # Load the vocabulary.
        vocabulary_path = os.path.join(config.dataset_path, "vocabulary.txt")
        assert os.path.exists(vocabulary_path), f"Error: {vocabulary_path} does not exist."
        self.vocabulary = open(vocabulary_path).read().split("\n")
        
        # Set the device. Need to be able to set this after the fact.
        self.device = None
        self.device_type = None

        self.data_loaders = {
            "train": None,
            "validate": None
        }


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

        # Get the data loader.
        data_loader = self.get_data_loader(split, shuffle, batch_size)

        # Yield each batch.
        for batch in data_loader:

            # Get the data.
            encoder_ids = batch[0]
            decoder_ids = batch[1]
            target_ids = batch[2]
            padding_masks = batch[3]

            # Move the data to the GPU.
            if self.device_type == "cuda":
                encoder_ids = encoder_ids.pin_memory().to(self.device, non_blocking=True)
                decoder_ids = decoder_ids.pin_memory().to(self.device, non_blocking=True)
                target_ids = target_ids.pin_memory().to(self.device, non_blocking=True)
                padding_masks = padding_masks.pin_memory().to(self.device, non_blocking=True)
            else:
                encoder_ids = encoder_ids.to(self.device)
                decoder_ids = decoder_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                padding_masks = padding_masks.to(self.device)

            # Yield the data.
            yield encoder_ids, decoder_ids, target_ids, padding_masks


    def get_data_loader(self, split:str, shuffle:bool, batch_size:int):

        # Get the data loader and return it if it already exists.
        if self.data_loaders[split] is not None:
            data_loader = self.data_loaders[split]
            return data_loader

        # Get the dataset.
        if split == "train":
            dataset = self.dataset_train
        elif split == "validate":
            dataset = self.dataset_validate
        else:
            raise Exception(f"Error: Unknown split {split}.")
        assert type(dataset) == datasets.arrow_dataset.Dataset, f"Error: {dataset} is not a valid dataset. Got {type(dataset)}."

        # Determine if we should do token dropout.
        do_token_dropout = False
        if self.config.token_dropout:
            if (split == "train" and self.config.token_dropout_training) or (split == "validate" and self.config.token_dropout_validation):
                do_token_dropout = True
        
        # If so get the mask token and the other relevant variables.
        do_token_dropout_encoder = False
        do_token_dropout_decoder = False
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

        def apply_token_dropout(ids):
            ids_augmented = []
            for id in ids:
                if np.random.rand() < token_dropout_probability:
                    ids_augmented.append(np.random.choice(mask_tokens))
                else:
                    ids_augmented.append(id)
            return ids_augmented

        # Define the collate function.
        def collate_fn(batch):

            # Create the lists.
            encoder_ids = []
            decoder_ids = []
            target_ids = []
            padding_masks = []

            # Iterate over the batch.
            for item in batch:

                # Get the data.
                encoder_ids.append(torch.tensor(item["encoder_ids"]))
                decoder_ids.append(torch.tensor(item["decoder_ids"]))
                target_ids.append(torch.tensor(item["target_ids"]))
                padding_masks.append(torch.tensor(item["padding_mask"]))

                # Apply token dropout.
                if do_token_dropout_encoder:
                    encoder_ids[-1] = apply_token_dropout(encoder_ids[-1])
                if do_token_dropout_decoder:
                    decoder_ids[-1] = apply_token_dropout(decoder_ids[-1])

            # Stack everything.
            encoder_ids = torch.stack(encoder_ids, dim=0).long()
            decoder_ids = torch.stack(decoder_ids, dim=0).long()
            target_ids = torch.stack(target_ids, dim=0).long()
            padding_masks = torch.stack(padding_masks, dim=0).float()

            # Return.
            return encoder_ids, decoder_ids, target_ids, padding_masks

        # Create a data loader.
        print(f"Create a data loader for split {split}...")
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True if split == "train" else False,
            num_workers=self.config.number_of_processes,
            pin_memory=True,
            drop_last=True,
        )

        self.data_loaders[split] = data_loader
        return data_loader

        
