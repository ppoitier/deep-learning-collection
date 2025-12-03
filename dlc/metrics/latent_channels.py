import torch
from torchmetrics import Metric


class LatentChannelVariance(Metric):
    """
    Computes the average variance of the encoder outputs per channel.
    Helps detect 'dead channels' in the latent space.
    """

    full_state_update: bool = False

    def __init__(self, embedding_dim: int):
        super().__init__()
        # Accumulate variance sum across batches
        self.add_state("variance_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_steps", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, z: torch.Tensor):
        # z: (B, C, H, W) or (B, C, T)
        # Calculate variance across the spatial/batch dims for this batch
        # We flatten everything except C: (B, C, H, W) -> (B*H*W, C) -> var(dim=0)

        # 1. Permute to put Channel last: (B, H, W, C)
        if z.ndim == 4:
            z_flat = z.permute(0, 2, 3, 1).flatten(0, 2)
        elif z.ndim == 3:  # (B, C, T)
            z_flat = z.permute(0, 2, 1).flatten(0, 1)
        else:
            raise ValueError(f"Unexpected input shape {z.shape}")

        # 2. Calculate variance per channel for this batch
        batch_var = torch.var(
            z_flat, dim=0
        ).mean()  # scalar: average variance of all channels

        self.variance_sum += batch_var
        self.n_steps += 1

    def compute(self):
        return self.variance_sum / self.n_steps
