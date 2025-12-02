from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn import functional as F


@dataclass
class QuantizerOutput:
    quantized_vectors: torch.Tensor
    quantized_indices: torch.Tensor
    loss: torch.Tensor


class Quantizer(nn.Module):
    def __init__(
            self,
            n_embeddings: int,
            embedding_dim: int,
            commitment_loss_factor: float,
            quantization_loss_factor: float,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_loss_factor = commitment_loss_factor
        self.quantization_loss_factor = quantization_loss_factor

        self.embeddings = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / self.n_embeddings, 1 / self.n_embeddings
        )

    def forward(self, z: Tensor):
        # Input z is expected to be (B, H, W, C) from the encoder
        original_shape = z.shape
        z_flat = z.reshape(-1, self.embedding_dim)

        # --- Find nearest neighbors ---
        # Calculate the squared Euclidean distance between input vectors and embeddings
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )

        # Find the closest embedding indices
        encoding_indices = torch.argmin(distances, dim=1)
        quantized_flat = self.embeddings(encoding_indices)
        quantized = quantized_flat.reshape(original_shape)

        # --- Calculate losses ---
        # The embedding loss (or codebook loss) updates the embedding vectors to match the encoder's output.
        embedding_loss = F.mse_loss(quantized, z.detach())
        # The commitment loss updates the encoder to produce outputs closer to the chosen embedding.
        commitment_loss = F.mse_loss(z, quantized.detach())

        loss = (
            self.quantization_loss_factor * embedding_loss
            + self.commitment_loss_factor * commitment_loss
        )

        # --- Straight-Through Estimator (STE) ---
        # Copy gradients from `quantized` to `z` in the backward pass.
        quantized = z + (quantized - z).detach()

        # --- Prepare output ---
        # Reshape to PyTorch's convention (B, C, H, W) for conv layers
        quantized_vectors = quantized.permute(0, 3, 1, 2).contiguous()
        # Reshape indices to match the spatial dimensions of the feature map
        quantized_indices = encoding_indices.reshape(
            original_shape[0], original_shape[1], original_shape[2]
        )

        return QuantizerOutput(
            quantized_vectors=quantized_vectors,
            quantized_indices=quantized_indices.unsqueeze(1),
            loss=loss,
        )


class QuantizerEMA(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        commitment_loss_factor: float,
        decay: float = 0.99,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.commitment_loss_factor = commitment_loss_factor

        # The codebook is a buffer, not a parameter, as it's updated manually.
        embeddings = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer("embeddings", embeddings)
        # Buffers to track the moving average of cluster sizes and assigned vectors.
        self.register_buffer("cluster_size", torch.zeros(n_embeddings))
        self.register_buffer("ema_embed", embeddings.clone())

    def forward(self, z: Tensor):
        # Input z is expected to be (B, H, W, C) from the encoder
        original_shape = z.shape
        z_flat = z.reshape(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embeddings).reshape(original_shape)

        # EMA update logic (only during training)
        if self.training:
            one_hot_encoding = F.one_hot(encoding_indices, self.n_embeddings).type(z_flat.dtype)

            # Update cluster sizes (how many vectors were assigned to each code)
            n_i = torch.sum(one_hot_encoding, dim=0)
            self.cluster_size = self.cluster_size * self.decay + n_i * (1 - self.decay)

            # Update the moving average of the vectors themselves
            dw = one_hot_encoding.t() @ z_flat
            ema_embed = self.ema_embed * self.decay + dw * (1 - self.decay)

            # Laplace smoothing to handle dead clusters
            n = torch.sum(self.cluster_size)
            self.cluster_size = (self.cluster_size + 1e-5) / (n + self.n_embeddings * 1e-5) * n

            self.embeddings.data.copy_(ema_embed / self.cluster_size.unsqueeze(-1))
            self.ema_embed.data.copy_(ema_embed)

        # The only loss is the commitment loss, to train the encoder.
        loss = self.commitment_loss_factor * F.mse_loss(z, quantized.detach())

        # --- Straight-Through Estimator (STE) ---
        # Copy gradients from `quantized` to `z` in the backward pass.
        quantized = z + (quantized - z).detach()

        # --- Prepare output ---
        # Reshape to PyTorch's convention (B, C, H, W) for conv layers
        quantized_vector = quantized.permute(0, 3, 1, 2).contiguous()
        # Reshape indices to match the spatial dimensions of the feature map
        quantized_indices = encoding_indices.reshape(
            original_shape[0], original_shape[1], original_shape[2]
        )

        return QuantizerOutput(
            quantized_vectors=quantized_vector,
            quantized_indices=quantized_indices.unsqueeze(1),
            loss=loss,
        )
