from dataclasses import dataclass

from torch import nn, Tensor
from torch.nn import functional as F
from dlc.vae.blocks import Decoder
from dlc.vq_vae.quantizer import Quantizer, QuantizerEMA
from dlc.vq_vae.blocks import VQ_Encoder


@dataclass
class VQVAE_Output:
    encoder_output: Tensor
    reconstructed_input: Tensor
    total_loss: Tensor
    reconstruction_loss: Tensor
    quantizer_loss: Tensor
    quantized_indices: Tensor


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 64,
        n_embeddings: int = 512,
        hidden_channels_enc: tuple = (128, 256),
        hidden_channels_dec: tuple = (256, 128),
        commitment_loss_factor: float = 0.25,
        quantization_loss_factor: float = 1.0,
        use_quantizer_ema: bool = False,
        quantizer_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings

        self.encoder = VQ_Encoder(in_channels, embedding_dim, hidden_channels_enc)
        if use_quantizer_ema:
            self.quantizer = QuantizerEMA(
                n_embeddings=n_embeddings,
                embedding_dim=embedding_dim,
                commitment_loss_factor=commitment_loss_factor,
                decay=quantizer_ema_decay,
            )
        else:
            self.quantizer = Quantizer(
                n_embeddings=n_embeddings,
                embedding_dim=embedding_dim,
                commitment_loss_factor=commitment_loss_factor,
                quantization_loss_factor=quantization_loss_factor,
            )
        self.decoder = Decoder(embedding_dim, in_channels, hidden_channels_dec)

    def loss_function(
        self,
        original_input: Tensor,
        reconstructed_input: Tensor,
        quantizer_loss: Tensor,
    ) -> tuple[Tensor, Tensor]:
        reconstruction_loss = F.mse_loss(reconstructed_input, original_input)
        total_loss = reconstruction_loss + quantizer_loss
        return total_loss, reconstruction_loss

    def forward(self, x: Tensor) -> VQVAE_Output:
        # 1. Encode the input image
        # Input: (B, C, H, W) -> Output: (B, E, H', W')
        z_e = self.encoder(x)

        # 2. Permute the dimensions for the quantizer
        # From: (B, E, H', W') -> To: (B, H', W', E)
        z_e_permuted = z_e.permute(0, 2, 3, 1).contiguous()

        # 3. Quantize the latent vectors
        quantizer_output = self.quantizer(z_e_permuted)

        # 4. Decode the quantized vectors to reconstruct the image
        # The quantizer_output.quantized_vector is already in (B, E, H', W') format
        x_hat = self.decoder(quantizer_output.quantized_vectors)

        # 4. Calculate losses
        total_loss, recon_loss = self.loss_function(x, x_hat, quantizer_output.loss)

        return VQVAE_Output(
            encoder_output=z_e,
            reconstructed_input=x_hat,
            total_loss=total_loss,
            reconstruction_loss=recon_loss,
            quantizer_loss=quantizer_output.loss,
            quantized_indices=quantizer_output.quantized_indices,
        )
