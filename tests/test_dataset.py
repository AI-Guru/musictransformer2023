import torch
import numpy as np
import sys
sys.path.append(".")
from source.dataset import DatasetConfig, Dataset

def main():

    test_dataset()

def test_dataset():

    # Create the dataset config and dataset.
    dataset_config = DatasetConfig(
        dataset_path = "data/jsfakes4bars/generation",
        token_dropout_mask_token = 666# "[TDR]"
    )
    dataset = Dataset(
        dataset_config,
    )
    print(dataset)

    # Get the first batch.
    batch = next(dataset.iterate("train", shuffle=False, batch_size=1))
    encoder_ids, decoder_ids, target_ids = batch
    print(f"encoder_ids: {encoder_ids.shape}")
    print(f"encoder_ids: {encoder_ids.numpy().tolist()}")
    print(f"decoder_ids: {decoder_ids.shape}")
    print(f"decoder_ids: {decoder_ids.numpy().tolist()}")
    print(f"target_ids: {target_ids.shape}")

if __name__ == '__main__':
    main()