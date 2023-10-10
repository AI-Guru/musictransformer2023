import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

@dataclass
class BaseTransformerConfig:

    block_size: int = 256 #1024
    vocab_size: int = 400 #50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4 #12
    n_head: int = 4# 12
    n_embd: int = 128 #768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    weight_sharing: bool = True # share weights between token and positional embeddings like GPT-2
    bottleneck: str = "none"
    bottleneck_channels_list: List[int] = field(default_factory=lambda: [192, 256, 320])
    flash_attention: bool = False


class BaseTransformer(nn.Module):

    def summary(self):
        summary = ""
        summary += f"Block size:        {self.get_block_size()}\n" 
        summary += f"Vocab size:        {self.get_vocab_size()}\n"  
        summary += f"Embedding size:    {self.get_embedding_size()}\n"
        summary += f"Principal shape:   {self.get_principal_shape()}\n" 
        summary += f"Bottleneck shape:  {self.get_bottleneck_shape()}\n" 
        summary += f"Compression ratio: {self.get_compression_ratio()}\n" 
        
        summary += "Bottleneck:\n"
        if self.bottleneck is not None:
            shapes = self.bottleneck.get_shapes()
            for name, shape in shapes:
                summary += f"{name}: {shape} ({np.prod(shape):,})\n"
            summary += f"Bottleneck parameters: {self.bottleneck.get_num_params():,}\n"

        return summary

    def get_block_size(self):
        return self.config.block_size
    
    def get_vocab_size(self):
        return self.config.vocab_size
    
    def get_embedding_size(self):
        return self.config.n_embd

    def get_compression_ratio(self):
        principal_shape = self.get_principal_shape()
        bottleneck_shape = self.get_bottleneck_shape()
        if bottleneck_shape is None:
            return None
        return np.prod(bottleneck_shape) / np.prod(principal_shape)
    
    def get_principal_shape(self):
        # Return the shape of the principal part of the model.
        return [self.config.block_size, self.config.n_embd]

    def get_bottleneck_shape(self):
        # Raise an exception if there is no bottleneck.
        if self.bottleneck is None:
            return None
        
        # Return the shape.
        return self.bottleneck.get_shape()    

    def load_generic(check_point_path, config_class, model_class):
        if not os.path.exists(check_point_path):
            raise Exception(f"Could not find checkpoint at {check_point_path}")
        # Load the checkpoint.
        checkpoint = torch.load(check_point_path, map_location="cpu")

        # Load the config.
        model_config_dict = checkpoint["model_config"]
        model_config = config_class(**model_config_dict)

        # Create the model.
        model = model_class(model_config)

        # Get the keys of the state dict. Sort them and make a list.
        state_dict_keys = model.state_dict().keys()
        state_dict_keys = sorted(state_dict_keys)

        # Get the keys of the loaded state dict. Sort them and make a list.
        loaded_state_dict_keys = checkpoint["model"].keys()
        loaded_state_dict_keys = sorted(loaded_state_dict_keys)

        # Get the keys that they don't have in common. XOR.
        missing_keys = set(state_dict_keys).symmetric_difference(loaded_state_dict_keys)
        if len(missing_keys) > 0:
            raise Exception(f"Missing {len(missing_keys)} keys: {missing_keys}")

        # Both lists should have the same length.
        assert len(state_dict_keys) == len(loaded_state_dict_keys), f"Length of state_dict_keys ({len(state_dict_keys)}) does not match length of loaded_state_dict_keys ({len(loaded_state_dict_keys)})"

        # Load the state dict into the model.
        model.load_state_dict(checkpoint["model"])

        # Return the model.
        return model