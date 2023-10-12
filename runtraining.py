import datetime
import sys
sys.path.append(".")
from source.trainer import TrainerConfig, Trainer
from source.transformer import TransformerConfig, Transformer
from source.encodertransformer import EncoderTransformerConfig, EncoderTransformer
from source.dataset import DatasetConfig, Dataset


def train():

    test = "test" in sys.argv

    # A vanilla Transformer.
    #config_class = TransformerConfig
    #model_class = Transformer

    # A BERT-like model.
    config_class = EncoderTransformerConfig
    model_class = EncoderTransformer

    # Get the timestamp as YYYYMMDD-HHMM.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Create the model config.
    model_config = config_class(
        
        vocab_size = 320,   # Number of tokens in the vocabulary.
        n_layer = 2,        # Number of transformer layers.
        n_head = 4,         # Number of attention heads.     
        n_embd = 128,       # Dimension of the embeddings.
        dropout = 0.0,      # Dropout probability.
        bias = False,       # Whether to use bias in the attention layers.
        block_size = 512,   # Maximum sequence length.

        weight_sharing = True, # Whether to share weights between the embeddings and the head.
        
        # Fully convolutional bottlenecks, either plain or variational.
        #bottleneck = "CNNBottleneck",
        bottleneck = "VariationalCNNBottleneck",
        bottleneck_channels_list=[128, 256],

        # Linear bottlenecks, either plain or variational. Transforms to 1D and then to back to 2D.
        #bottleneck = "Linear2DBottleneck",
        #bottleneck = "VariationalLinear2DBottleneck",
        #bottleneck_channels_list=[64, 16, 2],

        # Linear bottlenecks, either plain or variational. Applied on the embeddings.
        #bottleneck = "Linear1DBottleneck",
        #bottleneck = "VariationalLinear1DBottleneck",
        #bottleneck_channels_list=[1024, 128],

    )

    # Create the model.
    model = model_class(model_config)
    print(model.summary())
    if "summary" in sys.argv:
        return

    # Create the dataset config and dataset.
    dataset_config = DatasetConfig(
        #dataset_path = "data/lakhclean_mmmtrack_1bars_vae/generation", # The dataset path.
        dataset_path = "data/jsfakes4bars/generation", # The dataset path.
        token_dropout = False # No data augmentation.
    )
    dataset = Dataset(
        dataset_config,
    )
    print(dataset)

    # Create the Training config.
    trainer_config = TrainerConfig(

        # General settings.
        out_dir=f"output/lakhclean/encodertransformer_{timestamp}",
        batch_size=128,
        num_epochs=500, # 5
        device="auto",
        
        # Bottleneck config.
        bottleneck_loss_coefficient=0.0,
        bottleneck_loss_coefficient_max=0.1,
        bottleneck_loss_iterations=25_000,
        
        # Optimizer settings.
        #max_iters=50_000,   # Total number of training iterations.
        weight_decay=1e-1,  # Weight decay.
        beta1=0.9,          # Beta1 for Adam.
        beta2=0.95,         # Beta2 for Adam.
        grad_clip=1.0,      # Clip gradients at this value, or disable if == 0.0.

        # Learning rate decay settings.
        learning_rate=1e-4,     # Max learning rate.
        min_lr=1e-5,            # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla.
        decay_lr=True,          # Whether to decay the learning rate.
        warmup_iters=1_000,     # How many steps to warm up for.
        lr_decay_iters=50_000,  # Should be ~= max_iters per Chinchilla.

        # Wandb config.
        wandb_log=True,
        wandb_project="encodertransformer-vae-lakhclean",
        wandb_run_name=f"encodertransformer_{timestamp}",
        
        # When to evaluate.
        eval_every=2000,
        eval_mode="steps",

        # When to log.
        log_every=2000,
        log_mode="steps",

        # When to save.
        save_every=2000,
        save_mode= "steps",

        # Debugging.
        find_not_updated_layers=True,
        stop_on_vanishing_gradient=False,
        log_grad_norm=True,
    )

    # Create the trainer.
    trainer = Trainer(trainer_config)

    # Train.
    trainer.train(model, dataset)


# Run the training.
if __name__ == "__main__":
    train()