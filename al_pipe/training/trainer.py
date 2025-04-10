"""Training loop for the active learning framework.

This module implements the training loop used to train models in the active learning
pipeline. It handles model training, optimization, and logging of training metrics.
"""

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from al_pipe.data.base_dataset import BaseDataset


class Trainer:
    """Trainer class that handles model training and optimization."""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: The PyTorch model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimization
            epochs: Number of epochs to train for
            batch_size: Batch size for training
            **kwargs: Additional training parameters
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, train_data: BaseDataset) -> None:
        """
        Train the model on the provided data.

        Args:
            train_data: Dataset containing training samples and labels
        """
        self.model.train()
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_data, batch_labels in train_loader:
                self.optimizer.zero_grad()
                inputs = batch_data.to(self.device)
                labels = batch_labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
