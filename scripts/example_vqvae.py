from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import random_split, DataLoader

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from dlc.vq_vae.model import VQVAE
from dlc.trainers.vqvae import VQVAELightningModule


def launch_training():
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = ImageFolder(
        root="~/datasets/galaxy10/train", transform=transform
    )
    train_dataset, test_dataset = random_split(dataset, lengths=(0.8, 0.2))
    print("#train samples:", len(train_dataset))
    print("#test samples:", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=23)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=23)
    print("#train batches:", len(train_dataloader))
    print("#test batches:", len(test_dataloader))

    vq_vae = VQVAE(
        in_channels=3,
        embedding_dim=256,
        n_embeddings=512,
        hidden_channels_enc=(64, 128, 256, 512, 512),
        hidden_channels_dec=(512, 512, 256, 128, 64),
        commitment_loss_factor=0.25,
        quantization_loss_factor=1.0,
    )

    lightning_module = VQVAELightningModule(
        vq_vae=vq_vae,
        learning_rate=4e-4,
        n_warmup_epochs=20,
        plateau_patience=5,
        plateau_factor=0.5,
    )

    lightning_trainer = pl.Trainer(
        fast_dev_run=True,
        max_epochs=500,
        logger=TensorBoardLogger(save_dir="logs"),
    )

    lightning_trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )


if __name__ == "__main__":
    launch_training()
