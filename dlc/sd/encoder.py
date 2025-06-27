import torch
from torch import nn, Tensor
from torch.nn import functional as F
from decoder import VAE_ResidualBlock, VAE_AttentionBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (N, C_in, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (N, 128, H, W)
            VAE_ResidualBlock(128, 128),
            # (N, 128, H, W)
            VAE_ResidualBlock(128, 128),
            # (N, 128, H, W)
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            # (N, 128, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            # (N, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),
            # (N, 256, H/2, W/2)
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            # (N, 256, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            # (N, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),
            # (N, 512, H/4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            # (N, 512, H/8, W/8)
            VAE_AttentionBlock(512),
            # (N, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (N, 512, H/8, W/8)
            nn.GroupNorm(32, 512),
            # (N, 512, H/8, W/8)
            nn.SiLU(),
            # (N, 512, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (N, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
            # (N, 8, H/8, W/8)
        )

    def forward(self, x: Tensor, noise: Tensor) -> Tensor:
        # x: (N, C_in, H, W)
        # noise: (N, C_out, H/8, W/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # asymmetrical padding (right and bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        # (N, C_out, H/8, W/8) -> 2x(N, C_out/2, H/8, W/8)
        mean, log_variance = torch.chunk(x, chunks=2, dim=1)
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        variance = log_variance.exp()
        std = variance.sqrt()
        # Z~N(0, 1) -> N(mean, variance)
        x = mean + std * noise
        # Scaling constant (comes from the original repository)
        x *= 0.18215
        return x



