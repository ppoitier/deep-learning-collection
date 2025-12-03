import lightning as pl
from torch import optim
from torchmetrics import MeanMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid

from dlc.vq_vae.model import VQVAE, VQVAE_Output
from dlc.schedulers.warmup_on_plateau import WarmupReduceLROnPlateau
from dlc.metrics.codebook_distribution import CodebookStats
from dlc.metrics.latent_channels import LatentChannelVariance


class VQVAELightningModule(pl.LightningModule):
    def __init__(
            self,
            vq_vae: VQVAE,
            learning_rate: float = 1e-4,
            n_warmup_epochs: int = 10,
            plateau_patience: int = 5,
            plateau_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vq_vae'])
        self.vq_vae = vq_vae

        # Metrics for logging
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.train_quantizer_loss = MeanMetric()
        self.val_quantizer_loss = MeanMetric()

        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_codebook_stats = CodebookStats(n_embeddings=vq_vae.n_embeddings)
        self.val_latent_var = LatentChannelVariance(embedding_dim=vq_vae.embedding_dim)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        model_output = self.vq_vae(x)

        # Log metrics
        self.train_loss(model_output.total_loss)
        self.train_recon_loss(model_output.reconstruction_loss)
        self.train_quantizer_loss(model_output.quantizer_loss)

        self.log("train/total_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", self.train_recon_loss, on_step=False, on_epoch=True)
        self.log("train/quantizer_loss", self.train_quantizer_loss, on_step=False, on_epoch=True)

        return model_output.total_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out: VQVAE_Output = self.vq_vae(x)

        # Log metrics
        self.val_loss(out.total_loss)
        self.val_recon_loss(out.reconstruction_loss)
        self.val_quantizer_loss(out.quantizer_loss)

        # SSIM (Needs [0,1] range, assuming x is [-1, 1])
        x_norm = x * 0.5 + 0.5
        x_rec_norm = out.reconstructed_input * 0.5 + 0.5
        self.val_ssim(x_rec_norm, x_norm)
        self.val_codebook_stats(out.quantized_indices)
        self.val_latent_var(out.encoder_output)

        self.log("val/total_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", self.val_recon_loss, on_step=False, on_epoch=True)
        self.log("val/quantizer_loss", self.val_quantizer_loss, on_step=False, on_epoch=True)

        # Log reconstructed images once per epoch
        if batch_idx == 0:
            self._log_images(x, out.reconstructed_input)

    def on_validation_epoch_end(self):
        self.log("val/ssim", self.val_ssim.compute())
        codebook_stats = self.val_codebook_stats.compute()
        self.log_dict({f"val/codebook_{k}": v for k, v in codebook_stats.items()})
        self.log("val/latent_variance", self.val_latent_var.compute())

    def _log_images(self, original_imgs, reconstructed_imgs):
        # We only want to log a few images
        n_images = min(8, original_imgs.size(0))

        # Create a grid of original and reconstructed images
        # Un-normalize images from [-1, 1] to [0, 1] for visualization
        original_grid = make_grid(original_imgs[:n_images] * 0.5 + 0.5)
        reconstructed_grid = make_grid(reconstructed_imgs[:n_images] * 0.5 + 0.5)

        # Check if logger is available
        if self.logger:
            self.logger.experiment.add_image("val/original_images", original_grid, self.current_epoch)
            self.logger.experiment.add_image("val/reconstructed_images", reconstructed_grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = WarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=self.hparams.n_warmup_epochs,
            patience=self.hparams.plateau_patience,
            factor=self.hparams.plateau_factor,
            mode="min",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
