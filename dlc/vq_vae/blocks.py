from torch import nn, Tensor
from dlc.vae.blocks import ResidualBlock


class VQ_Encoder(nn.Module):
    """
    ### MODIFICATION HIGHLIGHTS ###
    - This is a new encoder class adapted for the VQ-VAE.
    - It's based on your original `Encoder` but modified to output a single tensor.
    - The final layers are changed to output `out_channels` (the embedding dimension)
      instead of `2 * out_channels` since we no longer need mean and log_variance.
    """
    def __init__(self, in_channels=3, out_channels=4, hidden_channels=(32, 64, 128)):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1)]
        for i in range(1, len(hidden_channels)):
            layers += [
                ResidualBlock(hidden_channels[i - 1], hidden_channels[i]),
                nn.Conv2d(hidden_channels[i], hidden_channels[i], kernel_size=3, stride=2, padding=1),
            ]
        # The final block now outputs `out_channels` directly
        layers += [
            nn.GroupNorm(32, hidden_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=3, padding=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C_in, H, W) -> (N, C_out, H', W')
        return self.layers(x)
