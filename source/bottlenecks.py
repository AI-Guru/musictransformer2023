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
    def __init__(self, block_size, n_embd):
        super(VariationalBottleneck, self).__init__()
        
        # Encoder layers
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(n_embd, n_embd // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(n_embd // 2, n_embd // 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # Produce mean and log variance for the latent space
        self.fc_mu = nn.Conv1d(n_embd // 4, n_embd // 8, kernel_size=5, stride=2, padding=2)
        self.fc_logvar = nn.Conv1d(n_embd // 4, n_embd // 8, kernel_size=5, stride=2, padding=2)
        
        # Decoder layers
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose1d(n_embd // 8, n_embd // 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_embd // 4, n_embd // 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_embd // 2, n_embd, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, return_loss=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        # Only return the KL loss.
        if return_loss:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return x_recon, kl_loss

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