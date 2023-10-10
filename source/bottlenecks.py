import torch
import torch.nn as nn
import sys
sys.path.append("..")

from source.layers import (
    TransposeLayer,
)

class BottleneckFactory:

    def get_class(class_name:str):
        the_class = getattr(sys.modules[__name__], class_name)
        return the_class
        

class BaseBottleneck(nn.Module):

    def __init__(self, config, encoder, decoder, mu, logvar, masking_layers=None):
        super(BaseBottleneck, self).__init__()

        self.block_size = config.block_size
        self.n_embd = config.n_embd

        self.encoder_layers = encoder
        self.decoder_layers = decoder
        self.mu = mu
        self.logvar = logvar
        self.masking_layers = masking_layers
    
    def forward(self, x, return_loss=False, padding_mask=None):
            
        # Encode the input.
        if self.mu is not None and self.logvar is not None:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
        else: 
            z = self.encode(x)

        # Decode.
        x_recon = self.decode(z)

        # Mask out the latent space.
        padding_mask_latent = None
        if padding_mask is not None and self.masking_layers is not None:
            padding_mask = padding_mask.unsqueeze(1)
            padding_mask_latent = self.masking_layers(padding_mask)
            #padding_mask_latent = padding_mask_latent.squeeze(1)

        # Return the reconstruction and the loss.
        if return_loss:
            if self.mu is not None and self.logvar is not None:
                kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                if padding_mask_latent is not None:
                    kl_loss = kl_loss * padding_mask_latent
                kl_loss = torch.sum(kl_loss)
                return x_recon, kl_loss
            else:
                return x_recon, None

        # Return the reconstruction.
        return x_recon
    
    def encode(self, x):
        h = self.encoder_layers(x)
        
        if self.mu is not None and self.logvar is not None:
            mu = self.mu(h)
            logvar = self.logvar(h)
            return mu, logvar
        else:
            return h
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        if self.mu is not None and self.logvar is not None:
            mu = self.mu(x)
            shapes += [(f"{self.mu} (mu)", mu.shape)]
            logvar = self.logvar(x)
            shapes += [(f"{self.logvar} (logvar)", logvar.shape)]

            # Reparameterize
            z = self.reparameterize(mu, logvar)
            shapes += [("z", z.shape)]
        else:
            z = x

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
        if self.mu is not None and self.logvar is not None:
            mu = self.mu(x)
            logvar = self.logvar(x)

            # Reparameterize
            z = self.reparameterize(mu, logvar)
        else:
            z = x

        # Return the shape of the latent space.
        return list(z.shape[1:])
    
    def get_num_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        return num_params
    

class Linear1DBottleneck(BaseBottleneck):

    # A layer that converts the samples to 1D, applies a linear layer, and converts back to 2D.

    def __init__(self, config):

        # Get the channel list.
        channels_list = [config.block_size * config.n_embd] + config.bottleneck_channels_list

        # Encoder layers.
        # Start with reshaping the input to 1D.
        # Then linear layers.
        encoder_layers = []
        encoder_layers += [nn.Flatten()]
        for i in range(len(channels_list) - 1):
            in_channels = channels_list[i]
            out_channels = channels_list[i + 1]
            encoder_layers.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            ])
        encoder = nn.Sequential(*encoder_layers)

        # Produce mean and log variance for the latent space
        mu = None
        logvar = None

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
        super(Linear1DBottleneck, self).__init__(config, encoder, decoder, mu, logvar)

class VariationalLinear1DBottleneck(BaseBottleneck):

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


class VariationalLinear2DBottleneck(BaseBottleneck):
    # Works linearly on the last dimension.

    def __init__(self, config):
            
        # Get the channel list.
        channels_list = [config.n_embd] + config.bottleneck_channels_list

        # Encoder layers.
        # Then linear layers.
        encoder_layers = []
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
        decoder_layers = []
        for i in range(len(channels_list) - 1, 0, -1):
            in_channels = channels_list[i]
            out_channels = channels_list[i - 1]
            decoder_layers.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            ])
        decoder = nn.Sequential(*decoder_layers)

        # Call the super constructor.
        super(VariationalLinear2DBottleneck, self).__init__(config, encoder, decoder, mu, logvar)


class Linear2DBottleneck(BaseBottleneck):
    # Works linearly on the last dimension.

    def __init__(self, config):
            
        # Get the channel list.
        channels_list = [config.n_embd] + config.bottleneck_channels_list

        # Encoder layers.
        # Then linear layers.
        encoder_layers = []
        for i in range(len(channels_list) - 1):
            in_channels = channels_list[i]
            out_channels = channels_list[i + 1]
            encoder_layers.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            ])
        encoder = nn.Sequential(*encoder_layers)

        # Produce mean and log variance for the latent space
        mu = None
        logvar = None

        # Decoder layers.
        decoder_layers = []
        for i in range(len(channels_list) - 1, 0, -1):
            in_channels = channels_list[i]
            out_channels = channels_list[i - 1]
            decoder_layers.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            ])
        decoder = nn.Sequential(*decoder_layers)

        # Call the super constructor.
        super(Linear2DBottleneck, self).__init__(config, encoder, decoder, mu, logvar)


class CNNBottleneck(BaseBottleneck):

    # A layer that converts the samples to 1D, applies a linear layer, and converts back to 2D.

    def __init__(self, config):

        # Get the channel list.
        channels_list = [config.n_embd] + config.bottleneck_channels_list

        # Encoder layers
        encoder_layers = []
        encoder_layers += [TransposeLayer(1, 2)]
        for channels_index in range(len(channels_list) - 1):
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
        mu = None
        logvar = None

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
        super(CNNBottleneck, self).__init__(config, encoder, decoder, mu, logvar, masking_layers)


class VariationalCNNBottleneck(BaseBottleneck):

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
                #nn.LeakyReLU(negative_slope=0.2, inplace=True)
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
                #nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        # Remove the last ReLU. TODO: Is this good?
        #decoder_layers = decoder_layers[:-1]
        decoder_layers += [TransposeLayer(1, 2)]
        decoder = nn.Sequential(*decoder_layers)

        # Call the super constructor.
        super(VariationalCNNBottleneck, self).__init__(config, encoder, decoder, mu, logvar, masking_layers)

        #self.apply(self._init_weights)

    def _init_weights(self, module):
        assert False

        # Use kaiming_normal_ for all conv layers.
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            pass

