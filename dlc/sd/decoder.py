import torch
from torch import nn, Tensor
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(3, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C_in, H, W)
        residue = x
        n, c, h, w = x.shape
        x = x.view(n, c, h * w)
        # x: (N, C_in, H*W)
        x = x.transpose(-1, -2)
        # x: (N, H*W, C)
        x = self.attention(x)
        # x: (N, H*W, C)
        x = x.transpose(-1, -2)
        # x: (N, C, H*W)
        x = x.view(n, c, h, w)
        # x: (N, C, H, W)
        x += residue
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        residue = x
        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (N, 4, H/8, W/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # (N, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (N, 512, H/8, W/8)
            nn.Upsample(scale_factor=2),
            # (N, 512, H/4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (N, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),
            # (N, 512, H/2, W/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # (N, 512, H/2, W/2)
            VAE_ResidualBlock(512, 256),
            # (N, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (N, 256, H/2, W/2)
            nn.Upsample(scale_factor=2),
            # (N, 256, H, W)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # (N, 256, H, W)
            VAE_ResidualBlock(256, 128),
            # (N, 128, H, W)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # (N, 128, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            # (N, 3, H, W)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 4, H/8, W/8)
        x /= 0.18215
        for module in self:
            x = module(x)
        return x
