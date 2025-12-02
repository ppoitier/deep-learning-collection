import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import Accuracy
import os


class ImageClassifier(L.LightningModule):
    """
    A PyTorch Lightning Module for image classification.
    It encapsulates the model, loss function, optimizers, and metrics.
    """
    def __init__(
            self,
            model: nn.Module,
            num_classes=10,
            learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def _prediction_step(self, batch):
        """Re-usable logic for computing loss and predictions."""
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        predictions = torch.argmax(logits, dim=1)
        return loss, predictions, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._prediction_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_accuracy(preds, targets)
        self.log('train_acc_step', self.train_accuracy, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_acc_epoch', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._prediction_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_accuracy(preds, targets)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


def launch_training(model, train_dataloader: DataLoader, val_dataloader: DataLoader, debug=False):
    lightning_module = ImageClassifier(model=model, num_classes=10, learning_rate=1e-3)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath='./checkpoints',
    #     filename='best-model-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=1,
    #     mode='min',
    # )

    logger = TensorBoardLogger("lightning_logs", name="galaxy10_cls")

    # Initialize a PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[early_stop_callback],
        accelerator='auto', # Use GPU if available, otherwise CPU
        log_every_n_steps=5,
        fast_dev_run=debug,
        # Set fast_dev_run=True to quickly check if the code runs without errors
        # fast_dev_run=True,
        # Set limit_train_batches and limit_val_batches for faster debugging
        # limit_train_batches=10,
        # limit_val_batches=5,
    )

    # Start the training loop
    print("Starting training...")
    trainer.fit(lightning_module, train_dataloader, val_dataloader)
    print("Training finished.")

    # # Optionally, load the best model
    # if os.path.exists(checkpoint_callback.best_model_path):
    #     print(f"Loading best model from: {checkpoint_callback.best_model_path}")
    #     best_model = ImageClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    #     return best_model
    # else:
    #     return model
