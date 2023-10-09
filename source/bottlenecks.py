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


class VariationalBottleneck(nn.Module):

    def __init__(self, config):
        super(VariationalBottleneck, self).__init__()

        self.block_size = config.block_size
        self.n_embd = config.n_embd

        channels_list = [config.n_embd] + config.bottleneck_channels_list

        # Encoder layers
        encoder_layers = []
        for channels_index in range(len(channels_list) - 2):
            in_channels = channels_list[channels_index]
            out_channels = channels_list[channels_index + 1]
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                nn.ReLU()
            ])
        self.encoder_layers = nn.Sequential(*encoder_layers)

        # Masking layer. This is used to mask out the latent space.
        masking_layers = []
        for i in range(len(channels_list) - 1):
            in_channels = 1
            out_channels = 1
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, padding_mode="replicate", bias=False)
            conv_layer.weight.data = torch.ones_like(conv_layer.weight.data) / 5
            conv_layer.weight.requires_grad = False
            masking_layers.append(conv_layer)
        self.masking_layers = nn.Sequential(*masking_layers)

        # Produce mean and log variance for the latent space
        in_channels = channels_list[-2]
        out_channels = channels_list[-1]
        self.fc_mu = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.fc_logvar = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)

        # Decoder layers
        decoder_layers = []
        for i in range(len(channels_list), 1, -1):
            in_channels = channels_list[i - 1]
            out_channels = channels_list[i - 2]
            decoder_layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU()
            ])
        self.decoder_layers = nn.Sequential(*decoder_layers)

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
        if padding_mask is not None:
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
        x = x.transpose(1, 2)
        h = self.encoder_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        y = self.decoder_layers(z)
        y = y.transpose(1, 2)
        return y
    
    def get_shapes(self, print_friendlier=True, activations=False):

        # Shapes of the encoder layers.
        shapes = []

        # A random vector of shape (batch_size, block_size, n_embd)
        x = torch.randn((1, self.block_size, self.n_embd))
        shapes += [("x", x.shape)]

        # Encode the input. Use the encoder, layer by layer.
        x = x.transpose(1, 2)
        shapes += [("x transposed", x.shape)]
        for layer in self.encoder_layers:
            x = layer(x)
            shapes += [(str(layer), x.shape)]

        # Produce mean and log variance for the latent space
        mu = self.fc_mu(x)
        shapes += [("mu", mu.shape)]
        logvar = self.fc_logvar(x)
        shapes += [("logvar", logvar.shape)]

        # Reparameterize
        z = self.reparameterize(mu, logvar)
        shapes += [("z", z.shape)]

        # Decode the latent space. Use the decoder, layer by layer.
        for layer in self.decoder_layers:
            z = layer(z)
            shapes += [(str(layer), z.shape)]
        y = z.transpose(1, 2)
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
        depth = len(self.encoder_layers) // 2  # since each layer is paired with a ReLU
        return (self.n_embd // (2 ** (depth + 1)), self.block_size // (2 ** depth))