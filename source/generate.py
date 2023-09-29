import os
import torch
from transformer import Transformer
from tokenizer import Tokenizer


def main(
    output_path = "out"
):

    # Load the tokenizer.
    tokenizer_path = os.path.join(output_path, "tokenizer.json")
    tokenizer = Tokenizer.from_config_file(tokenizer_path)
    print(tokenizer.vocabulary_size)
    print(tokenizer.vocabulary)

    # Load the model.
    checkpoint_path = os.path.join(output_path, "ckpt.pt")
    model = Transformer.load(checkpoint_path)
    print(model)

    # Sample from a normal distribution.
    bottleneck_shape = model.get_bottleneck_shape()
    bottleneck_z = torch.randn(1, *bottleneck_shape)
    print(bottleneck_z.shape)

    for _ in range(2):
        # Create the start sequence.
        start_sequence = "PIECE_START"
        start_sequence_indices = tokenizer.encode_sequence(start_sequence)
        print(f"Start sequence: {start_sequence}")
        print(f"Start sequence indices: {start_sequence_indices}")

        result_ids = model.generate(
            idx=start_sequence_indices,
            max_new_tokens=10,
            bottleneck_condition=bottleneck_z, 
            temperature=0.5,
            top_k=None
        )[0]
        print(f"Result ids: {result_ids}")

        result_sequence = tokenizer.decode_sequence(result_ids)
        print(f"Result sequence: {result_sequence}")


if __name__ == "__main__":
    main()