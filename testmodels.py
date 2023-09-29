import torch
from nanoGPT.model import GPTConfig, GPT
from source.transformer import TransformerConfig, Transformer

# Create the model.
print("GPT")
config = GPTConfig()
config.block_size = 256
config.vocab_size = 1000
config.n_layer = 4
config.n_head = 4
config.n_embd = 128
model = GPT(config)

# Do a forward pass.
x = torch.randint(0, 1000, (1, 10))
print(f"x.shape: {x.shape}")
y, loss = model(x)
print(f"y.shape: {y.shape}")

print("")

# Create the model
print("Transformer")
config = TransformerConfig()
config.block_size = 256
config.vocab_size = 1000
config.n_layer = 4
config.n_head = 4
config.n_embd = 128
model = Transformer(config)

# Do a forward pass.
x = torch.randint(0, 1000, (1, 10))
print(f"x.shape: {x.shape}")
y, loss = model(x, x)
print(f"y.shape: {y.shape}")