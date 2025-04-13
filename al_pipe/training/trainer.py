"""Training loop for the active learning framework.

This module implements the training loop used to train models in the active learning
pipeline using PyTorch Lightning. It handles model training, optimization, and logging
of training metrics.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder


class Trainer(pl.LightningModule):
    """Trainer class that handles model training and optimization using PyTorch Lightning."""

    def __init__(
        self,
        model: BaseStaticEmbedder | nn.Module,  # TODO: This is a hack to allow for both static and trainable models
        learning_rate: float = 0.001,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: The PyTorch model to train
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            **kwargs: Additional training parameters
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Model output tensor
        """
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for Lightning.

        Args:
            batch: Tuple of (inputs, labels)
            batch_idx: Index of current batch

        Returns:
            Loss tensor
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            PyTorch optimizer
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self) -> DataLoader:
        """
        Configure the training DataLoader.

        Returns:
            PyTorch DataLoader for training
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def train(self, train_data: BaseDataset, max_epochs: int = 10) -> None:
        """
        Train the model on the provided data.

        Args:
            train_data: Dataset containing training samples and labels
            max_epochs: Number of epochs to train for
        """
        self.train_data = train_data
        trainer = pl.Trainer(max_epochs=max_epochs)
        trainer.fit(self)
