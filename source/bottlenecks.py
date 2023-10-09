import torch
import torch.nn as nn

    
class SimpleBottleneck(nn.Module):
    def __init__(self, block_size, n_embd):
        super(SimpleBottleneck, self).__init__()
        
        # Encoder
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(n_embd, n_embd // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_embd // 2, n_embd // 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_embd // 4, n_embd // 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose1d(n_embd // 8, n_embd // 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_embd // 4, n_embd // 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_embd // 2, n_embd, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )
    
    def forward(self, x, return_loss=False):
        x = self.encode(x)
        x = self.decode(x)
        if return_loss:
            return x, 0.0
        return x

    def encode(self, x):
        x = x.transpose(1, 2)
        z = self.encoder_layers(x)
        return z

    def decode(self, z):
        y = self.decoder_layers(z)
        y = y.transpose(1, 2)
        return y


class BaseVariationalBottleneck(nn.Module):

    # A layer that converts the samples to 1D, applies a linear layer, and converts back to 2D.

    def __init__(self, config, encoder, decoder, mu, logvar, masking_layers=None):
        super(BaseVariationalBottleneck, self).__init__()

        self.block_size = config.block_size
        self.n_embd = config.n_embd

        self.encoder_layers = encoder
        self.decoder_layers = decoder
        self.mu = mu
        self.logvar = logvar
        self.masking_layers = masking_layers

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, return_loss=False, padding_mask=None):
            
        # Note: Shape is (batch_size, block_size, n_embd) as it comes from the transformer.

        # Encode the input.
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        # Mask out the latent space.
        padding_mask_latent = None
        if padding_mask is not None and self.masking_layers is not None:
            padding_mask = padding_mask.unsqueeze(1)
            padding_mask_latent = self.masking_layers(padding_mask)
            #padding_mask_latent = padding_mask_latent.squeeze(1)

        # Return the reconstruction and the loss.
        if return_loss:
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            #print(f"kl_loss.shape: {kl_loss.shape}")
            if padding_mask_latent is not None:
                #print(f"padding_mask_latent.shape: {padding_mask_latent.shape}")
                #print(f"padding_mask_latent: {padding_mask_latent}")
                kl_loss = kl_loss * padding_mask_latent
                #print(f"kl_loss.shape: {kl_loss.shape}")
                #print(f"kl_loss: {kl_loss.detach().transpose(2, 1).numpy().tolist()}")
            kl_loss = torch.sum(kl_loss)
            return x_recon, kl_loss

        # Return the reconstruction.
        return x_recon
    
    def encode(self, x):
        h = self.encoder_layers(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
    def decode(self, z):
        y = self.decoder_layers(z)
        return y
    
    def get_shapes(self, print_friendlier=True, activations=False):
            
        # Shapes of the encoder layers.
        shapes = []

        # A random vector of shape (batch_size, block_size, n_embd)
        x = torch.randn((1, self.block_size, self.n_embd))
        shapes += [("x", x.shape)]

        # Encode the input. Use the encoder, layer by layer.
        for layer in self.encoder_layers:
            x = layer(x)
            shapes += [(str(layer), x.shape)]

        # Produce mean and log variance for the latent space
        mu = self.mu(x)
        shapes += [(f"{self.mu} (mu)", mu.shape)]
        logvar = self.logvar(x)
        shapes += [(f"{self.logvar} (logvar)", logvar.shape)]

        # Reparameterize
        z = self.reparameterize(mu, logvar)
        shapes += [("z", z.shape)]

        # Decode the latent space. Use the decoder, layer by layer.
        for layer in self.decoder_layers:
            z = layer(z)
            shapes += [(str(layer), z.shape)]
        y = z
        shapes += [("y", y.shape)]

        # Filter out the activations.
        if activations == False:
            shapes = [shape for shape in shapes if not shape[0].startswith("ReLU")]

        # If print friendlier, make sure that all the layer names have the same length - add spaces.
        # Also map shapes to tuples.
        if print_friendlier:
            max_name_length = max([len(name) for name, shape in shapes])
            shapes = [(name + " " * (max_name_length - len(name)), shape) for name, shape in shapes]
            shapes = [(name, list(shape)) for name, shape in shapes]

        return shapes
    
    def get_shape(self):
        # A random vector of shape (batch_size, block_size, n_embd)
        x = torch.randn((1, self.block_size, self.n_embd))

        # Run it through the encoder.
        x = self.encoder_layers(x)

        # Run through the mean and log variance.
        mu = self.mu(x)
        logvar = self.logvar(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Return the shape of the latent space.
        return list(z.shape[1:])
    
    def get_num_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        return num_params
    

class VariationalLinear1DBottleneck(BaseVariationalBottleneck):

    # A layer that converts the samples to 1D, applies a linear layer, and converts back to 2D.

    def __init__(self, config):

        # Get the channel list.
        channels_list = [config.block_size * config.n_embd] + config.bottleneck_channels_list

        # Encoder layers.
        # Start with reshaping the input to 1D.
        # Then linear layers.
        encoder_layers = []
        encoder_layers += [nn.Flatten()]
        for i in range(len(channels_list) - 2):
            in_channels = channels_list[i]
            out_channels = channels_list[i + 1]
            encoder_layers.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            ])
        encoder = nn.Sequential(*encoder_layers)

        # Produce mean and log variance for the latent space
        in_channels = config.bottleneck_channels_list[-2]
        out_channels = config.bottleneck_channels_list[-1]
        mu = nn.Linear(in_channels, out_channels)
        logvar = nn.Linear(in_channels, out_channels)

        # Decoder layers.
        # Start with linear layers.
        # Then reshape back to 2D.
        decoder_layers = []
        for i in range(len(channels_list) - 1, 0, -1):
            in_channels = channels_list[i]
            out_channels = channels_list[i - 1]
            decoder_layers.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            ])
        decoder_layers += [nn.Unflatten(1, (config.block_size, config.n_embd))]
        decoder = nn.Sequential(*decoder_layers)

        # Call the super constructor.
        super(VariationalLinear1DBottleneck, self).__init__(config, encoder, decoder, mu, logvar)

class TransposeLayer(nn.Module):

    def __init__(self, dim1, dim2):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class VariationalCNNBottleneck(BaseVariationalBottleneck):

    # A layer that converts the samples to 1D, applies a linear layer, and converts back to 2D.

    def __init__(self, config):

        # Get the channel list.
        channels_list = [config.n_embd] + config.bottleneck_channels_list

        # Encoder layers
        encoder_layers = []
        encoder_layers += [TransposeLayer(1, 2)]
        for channels_index in range(len(channels_list) - 2):
            in_channels = channels_list[channels_index]
            out_channels = channels_list[channels_index + 1]
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                nn.ReLU()
            ])
        encoder = nn.Sequential(*encoder_layers)

        # Masking layer. This is used to mask out the latent space.
        masking_layers = []
        for i in range(len(channels_list) - 1):
            in_channels = 1
            out_channels = 1
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, padding_mode="replicate", bias=False)
            conv_layer.weight.data = torch.ones_like(conv_layer.weight.data) / 5
            conv_layer.weight.requires_grad = False
            masking_layers.append(conv_layer)
        masking_layers = nn.Sequential(*masking_layers)

        # Produce mean and log variance for the latent space
        in_channels = channels_list[-2]
        out_channels = channels_list[-1]
        mu = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        logvar = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)

        # Decoder layers
        decoder_layers = []
        for i in range(len(channels_list), 1, -1):
            in_channels = channels_list[i - 1]
            out_channels = channels_list[i - 2]
            decoder_layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU()
            ])
        decoder_layers += [TransposeLayer(1, 2)]
        decoder = nn.Sequential(*decoder_layers)

        # Call the super constructor.
        super(VariationalCNNBottleneck, self).__init__(config, encoder, decoder, mu, logvar, masking_layers)
