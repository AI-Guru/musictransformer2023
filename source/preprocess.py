# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset # huggingface datasets
import fire
from collections import Counter
import json
import matplotlib.pyplot as plt

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

def peprocess(
    dataset_id="TristanBehrens/js-fakes-4bars",
    output_path="data/jsfakes4bars",
    padding_length=384,
    mode="generation" # "modeling"
    ):

    # Create the output path.
    output_path = os.path.join(output_path, mode)
    os.makedirs(output_path, exist_ok=True)

    # Open the overview file.
    overview_file_path = os.path.join(output_path, "overview.txt")
    overview_file = open(overview_file_path, "w")

    # Load the dataset.
    print(f"Loading dataset {dataset_id}...")
    split_dataset = load_dataset(dataset_id)
    split_dataset["val"] = split_dataset.pop("test")
    for split in split_dataset.keys():
        print(f"{split}: {split_dataset[split].num_rows} rows")

    # Check if we can work with this dataset.
    required_keys = ["train", "val"]
    for required_key in required_keys:
        if required_key not in split_dataset.keys():
            print(f"Error: {required_key} not in split_dataset.keys()")

    # Get the vocabulary from the dataset. Use a counter and iterate over the dataset.
    # This is not the most efficient way to do this, but it works.
    print("Getting vocabulary...")
    vocabulary_counter = Counter()
    for split in split_dataset.keys():
        for row in tqdm(split_dataset[split]):
            vocabulary_counter.update(row["text"].split(" "))
    vocabulary = [word for word, _ in vocabulary_counter.most_common()]
    vocabulary = sorted(vocabulary)

    # We need to add the special tokens.
    vocabulary = ["[UNK]", "[PAD]", "[SOS]", "[EOS]", "[TDR]"] + vocabulary

    # Save the vocabulary.
    vocabulary_file_path = os.path.join(output_path, "vocabulary.txt")
    with open(vocabulary_file_path, "w") as vocabulary_file:
        for token in vocabulary:
            vocabulary_file.write(f"{token}\n")

    # Print some information.
    print(f"Vocabulary size: {len(vocabulary)}", file=overview_file)

    # A method that encodes a text.
    def encode_text(text, raise_exception_on_unk=True):
        encoded_text = [encode_token(token) for token in text.split(" ")]
        if raise_exception_on_unk:
            assert encode_token("[UNK]") not in encoded_text
        return encoded_text

    def encode_token(token):
        index = vocabulary.index(token) if token in vocabulary else encode_token("[UNK]")
        #assert index != encode_token("[UNK]")
        return index
    
    # Test the encoding.
    print("Testing the encoding...")
    text = "PIECE_START TRACK_START"
    print(f"Text: {text}", file=overview_file)
    encoded_text = encode_text(text)
    print(f"Encoded text: {encoded_text}", file=overview_file)

    # Now we want to tokenize the dataset.
    def process(example):
        # Encode the text.
        text_ids = encode_text(example["text"])
        assert encode_token("[UNK]") not in text_ids

        # Create the ids.
        if mode == "modeling":
            encoder_ids = text_ids
            decoder_ids = [encode_token("[SOS]")] + text_ids
            target_ids = text_ids + [encode_token("[EOS]")]
        elif mode == "generation":
            encoder_ids = text_ids
            decoder_ids = text_ids
            target_ids = text_ids[1:] + [encode_token("[EOS]")]

        # Raise an exception if any of the sequences is longer than the padding length.
        if len(encoder_ids) > padding_length:
            raise Exception(f"Encoder sequence too long! It has length {len(encoder_ids)}")
        if len(decoder_ids) > padding_length:
            raise Exception(f"Decoder sequence too long! It has length {len(decoder_ids)}")
        if len(target_ids) > padding_length:
            raise Exception(f"Target sequence too long! It has length {len(target_ids)}")

        # Pad the ids. Add padding token at the end.
        encoder_ids += [encode_token("[PAD]")] * (padding_length - len(encoder_ids))
        decoder_ids += [encode_token("[PAD]")] * (padding_length - len(decoder_ids))
        target_ids += [encode_token("[PAD]")] * (padding_length - len(target_ids))

        out = {
            "encoder_ids": encoder_ids,
            "decoder_ids": decoder_ids,
            "target_ids": target_ids,
            "length": len(text_ids),
        }   
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        #num_proc=num_proc,
    )

    # Print a sample.
    print("Sample:", file=overview_file)
    print(f"encoder_ids: {tokenized['train'][0]['encoder_ids']}", file=overview_file)
    print(f"decoder_ids: {tokenized['train'][0]['decoder_ids']}", file=overview_file)
    print(f"target_ids:  {tokenized['train'][0]['target_ids']}", file=overview_file)

    # Create a histogram of the lengths.
    lengths = [row["length"] for row in tokenized["train"]]
    plt.hist(lengths, bins=100)
    plt.savefig(os.path.join(output_path, "lengths_distribution.png"))

    # Write the information to the overview file.
    print(f"Min length: {np.min(lengths)}", file=overview_file)
    print(f"Max length: {np.max(lengths)}", file=overview_file)
    print(f"Mean length: {np.mean(lengths)}", file=overview_file)
    print(f"Std length: {np.std(lengths)}", file=overview_file)

    # Save the dataset as JSONL files. One for each split.
    print("Saving the dataset...")
    for split in tokenized.keys():
        split_file_path = os.path.join(output_path, f"{split}.jsonl")
        with open(split_file_path, "w") as split_file:
            for row in tqdm(tokenized[split]):
                json.dump(row, split_file)
                split_file.write("\n")


if __name__ == "__main__":
    fire.Fire(peprocess)