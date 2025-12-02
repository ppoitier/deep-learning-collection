import torch
from torch import nn, Tensor
from dlc.vae.blocks import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, hidden_channels_enc=(128, 64, 32), hidden_channels_dec=(128, 64, 32)):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, hidden_channels_enc)
        self.decoder = Decoder(latent_dim, in_channels, hidden_channels_dec)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std)
        z = mean + noise * std
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
