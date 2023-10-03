# musictransformer2023

# Goal.

The goal is to create a music transformer that has a variational bottleneck. Usecases would be musical interpolation and generation of long sequences.

# Notes.

- This is an early work in progress. The essential pieces are there but the model is not yet good.
- The architecture is a classic encoder-decoder transformer. See "Attention is all you need" for details.
- The variational bottleneck is a fully convolutional network. Instead of a vector, the latent space is a matrix. This is inspired by latent diffusion.
- I am looking for a way to weaken the decoder. If the decoder is too strong the latent space is not taken into account.
- Currently I am experimenting with token dropout to weaken the decoder. VAE beta warmup is also an option, but not implemented yet.
- Currently I am experimenting with 500 midi files from the js-fakes dataset. This allows me to train on a single GPU. 
- It is planned to go to a bigger dataset soon.

# Getting started.

- Prepare to submit Github issues. I am looking for collaborators. 🤗 https://github.com/AI-Guru/musictransformer2023/issues
- Get a decent GPU. I am using an A100, which is clearly overkill. Way smaller GPUs will do just fine.
- Create a dataset by running `python source/preprocess.py`. This will download the js-fakes dataset and prepare it for training.
- Better set up Weights and Biases. Create an account and run `wandb login`. This will allow you to track your experiments. https://wandb.ai/
- Run `python runtraining.py`. This will train the model. You can track the progress on Weights and Biases.
- Note `python runtraininggrid.py` is an example for grid search.

# Acknowledgements.

- This repo is based on Andrej Karpathy's nanoGPT. https://github.com/karpathy/nanoGPT
