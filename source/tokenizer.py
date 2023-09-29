from dataclasses import dataclass, field, asdict
import json


@dataclass
class TokenizerConfig:
    vocabulary_size: int = 0
    vocabulary: list = field(default_factory=list) 

class Tokenizer:

    def __init__(self, config: TokenizerConfig):
        self.vocabulary_size = config.vocabulary_size
        self.vocabulary = config.vocabulary

    def from_vocabulary_file(vocabulary_file):
        # Load the vocabulary file and split it.
        with open(vocabulary_file, "r") as vocabulary_file:
            vocabulary = vocabulary_file.read().split("\n")
        return Tokenizer.from_vocabulary(vocabulary)
        
    def from_vocabulary(vocabulary):
        config = TokenizerConfig(
            vocabulary_size=len(vocabulary),
            vocabulary=vocabulary
        )
        return Tokenizer(config)
    
    def from_config_file(config_file):
        config_dict = json.load(open(config_file, "r"))
        config = TokenizerConfig(**config_dict)
        return Tokenizer(config)

    def save(self, path:str):
        config = TokenizerConfig(
            vocabulary_size=self.vocabulary_size,
            vocabulary=self.vocabulary
        )
        # Map to dict and save JSON.
        config_dict = asdict(config)
        with open(path, "w") as config_file:
            json.dump(config_dict, config_file, indent=4)


    def encode_sequence(self, sequence):
        if isinstance(sequence, str):
            sequence = sequence.split(" ")
        assert isinstance(sequence, list)

        sequence_encoded = [self.encode_token(token) for token in sequence]
        return sequence_encoded

    def encode_token(self, token):
        index = self.vocabulary.index(token) if token in self.vocabulary else self.encode_token("[UNK]")
        return index

    def decode_sequence(self, ids):
        sequence = [self.decode_token(id) for id in ids]
        return sequence
    
    def decode_token(self, id):
        token = self.vocabulary[id]
        return token