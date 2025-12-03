import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import lightning as pl
from torchmetrics.classification import MulticlassAccuracy

from dlc.vq_vae.model import VQVAE, VQVAE_Output

class MaskGITLightningModule(pl.LightningModule):
    def __init__(
            self,
            transformer: nn.Module,
            vq_vae: VQVAE,
            learning_rate: float = 1e-4,
            mask_token_id: int = 1024,
            vocab_size: int = 1024,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transformer", "vq_vae"])

        self.transformer = transformer
        self.vq_vae = vq_vae
        self.vq_vae.eval()
        self.mask_token_id = mask_token_id

        # Metrics: Accuracy on the MASKED tokens only
        self.train_acc = MulticlassAccuracy(num_classes=vocab_size)
        self.val_acc = MulticlassAccuracy(num_classes=vocab_size)

    def _compute_cosine_mask_ratio(self, batch_size, device):
        """
        Samples the mask ratio 'gamma' from a cosine distribution.
        Paper: gamma(r) = cos(r * pi / 2) where r ~ U(0, 1)
        """
        # r ~ Uniform(0, 1)
        r = torch.rand(batch_size, device=device)
        # gamma ~ Cosine Schedule (favors higher mask ratios)
        gamma = torch.cos(r * math.pi * 0.5)
        return gamma

    def _apply_masking(self, x_indices):
        """
        Vectorized Random Masking.
        x_indices: (B, T)
        Returns: masked_indices (B, T), mask_mask (B, T)
        """
        B, T = x_indices.shape
        device = x_indices.device

        # 1. Determine how many tokens to mask for each sample
        gamma = self._compute_cosine_mask_ratio(B, device) # (B,)
        n_masks = torch.floor(gamma * T).long() # (B,)

        # 2. Generate random noise to select indices
        noise = torch.rand((B, T), device=device)

        # 3. Find the 'cutoff' rank for each sample
        # We sort the noise. The indices corresponding to the smallest noise values get masked.
        # This is efficient vectorized "sampling without replacement"
        sorted_indices = torch.argsort(noise, dim=1) # (B, T)
        ranks = torch.argsort(sorted_indices, dim=1) # (B, T) returns the rank of each position

        # Create mask: True if rank < n_masks
        # unsqueeze n_masks to broadcast: (B, 1) vs (B, T)
        mask_bool = ranks < n_masks.unsqueeze(1)

        # 4. Apply Mask
        x_masked = x_indices.clone()
        x_masked[mask_bool] = self.mask_token_id

        return x_masked, mask_bool

    def training_step(self, batch, batch_idx):
        # 1. Get Image
        x, _ = batch # (B, C, H, W)

        # 2. Tokenize with Frozen VQ-VAE
        with torch.no_grad():
            out: VQVAE_Output = self.vq_vae(x)
            target_indices = out.quantized_indices.flatten(1).detach()

        # 3. Apply Masking Strategy
        x_masked, mask_bool = self._apply_masking(target_indices)

        # 4. Transformer Forward Pass
        # Input: (B, T) -> Output: (B, T, Vocab_Size)
        logits = self.transformer(x_masked)

        # 5. Compute Loss ONLY on Masked Tokens
        # Cross Entropy expects (B, C, T), so we permute logits
        loss = F.cross_entropy(logits.transpose(1, 2), target_indices, reduction='none')

        # Filter loss: Only gradients for masked positions matter
        # We normalize by the number of masked tokens to keep loss magnitude stable
        masked_loss = (loss * mask_bool.float()).sum() / (mask_bool.sum() + 1e-6)

        # 6. Metrics
        # Compute accuracy only on the masked predictions
        # Flatten for metric: (B*T_masked, Vocab)
        if mask_bool.sum() > 0:
            flat_logits = logits[mask_bool]
            flat_targets = target_indices[mask_bool]
            self.train_acc(flat_logits, flat_targets)
            self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train/loss", masked_loss, on_step=True, on_epoch=True, prog_bar=True)
        return masked_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad():
            out: VQVAE_Output = self.vq_vae(x)
            target_indices = out.quantized_indices.flatten(1).detach()
        # In validation, we might want to test a fixed mask ratio (e.g., 50%)
        # or use the random schedule to see general robustness.
        # Using random schedule here for consistency with 'train loss'
        x_masked, mask_bool = self._apply_masking(target_indices)

        logits = self.transformer(x_masked)
        loss = F.cross_entropy(logits.transpose(1, 2), target_indices, reduction='none')
        masked_loss = (loss * mask_bool.float()).sum() / (mask_bool.sum() + 1e-6)

        if mask_bool.sum() > 0:
            self.val_acc(logits[mask_bool], target_indices[mask_bool])
            self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)

        self.log("val/loss", masked_loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.transformer.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.05
        )
        # Cosine Annealing is standard for Transformers
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
