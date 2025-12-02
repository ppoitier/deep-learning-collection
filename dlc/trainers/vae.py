import lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
from dlc.schedulers.warmup_on_plateau import WarmupReduceLROnPlateau


class VAELightningModule(pl.LightningModule):

    def __init__(
        self,
        vae: nn.Module,
        learning_rate=1e-4,
        kl_coef=0.00025,
        n_warmup_epochs=10,
        plateau_patience=5,
        plateau_factor=0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.vae = vae

        # Metrics for logging
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.train_kl_loss = MeanMetric()
        self.val_kl_loss = MeanMetric()

    def compute_loss(self, x, x_hat, mean, log_var):
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        # KL Divergence Loss
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # We take the mean over the batch dimension
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=[1, 2, 3])
        kl_loss = torch.mean(kl_loss, dim=0)
        total_loss = recon_loss + self.hparams.kl_coef * kl_loss
        return total_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mean, log_var = self.vae(x)
        total_loss, recon_loss, kl_loss = self.compute_loss(x, x_hat, mean, log_var)

        # Log metrics
        self.train_loss(total_loss)
        self.train_recon_loss(recon_loss)
        self.train_kl_loss(kl_loss)

        self.log(
            "train/total_loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/recon_loss", self.train_recon_loss, on_step=False, on_epoch=True
        )
        self.log("train/kl_loss", self.train_kl_loss, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mean, log_var = self.vae(x)
        total_loss, recon_loss, kl_loss = self.compute_loss(x, x_hat, mean, log_var)

        # Log metrics
        self.val_loss(total_loss)
        self.val_recon_loss(recon_loss)
        self.val_kl_loss(kl_loss)

        self.log(
            "val/total_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/recon_loss", self.val_recon_loss, on_step=False, on_epoch=True)
        self.log("val/kl_loss", self.val_kl_loss, on_step=False, on_epoch=True)

        # Log reconstructed images once per epoch
        if batch_idx == 0:
            self._log_images(x, x_hat)

    def _log_images(self, original_imgs, reconstructed_imgs):
        # We only want to log a few images
        n_images = min(8, original_imgs.size(0))

        # Create a grid of original and reconstructed images
        # Un-normalize images from [-1, 1] to [0, 1] for visualization
        original_grid = make_grid(original_imgs[:n_images] * 0.5 + 0.5)
        reconstructed_grid = make_grid(reconstructed_imgs[:n_images] * 0.5 + 0.5)

        self.logger.experiment.add_image(
            "val/original_images", original_grid, self.current_epoch
        )
        self.logger.experiment.add_image(
            "val/reconstructed_images", reconstructed_grid, self.current_epoch
        )

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
