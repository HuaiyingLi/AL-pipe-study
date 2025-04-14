"""Evaluator module for assessing model performance during active learning.

This module provides functionality to evaluate embedding models during the active learning
process using PyTorch Lightning, tracking various metrics like loss, accuracy, and other
relevant performance indicators.
"""

import pytorch_lightning as pl
import torch
import torchmetrics

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder
from al_pipe.util.general import avail_device


class Evaluator(pl.LightningModule):
    """Evaluator class for assessing model performance during active learning iterations."""

    def __init__(
        self,
        metrics: dict[str, any] | None = None,
        batch_size: int = 32,
        device: str = "cuda",
    ) -> None:
        """Initialize the evaluator.

        Args:
            metrics: Dictionary of metric functions to compute during evaluation.
                If None, defaults to basic metrics like MSE loss.
            batch_size: Batch size for evaluation.
            device: Device to run evaluation on ('cuda' or 'cpu').
        """
        super().__init__()
        self.batch_size = batch_size
        self.device = avail_device(device)

        # Initialize metrics using torchmetrics
        self.metrics = (
            torch.nn.ModuleDict(
                {
                    "mse": torchmetrics.MeanSquaredError(),
                    "mae": torchmetrics.MeanAbsoluteError(),
                    "r2": torchmetrics.R2Score(),
                }
            )
            if metrics is None
            else metrics
        )

        # Save hyperparameters
        self.save_hyperparameters(ignore=["metrics"])

    def evaluate(
        self, embed_model: BaseStaticEmbedder, model: torch.nn.Module, dataloader: BaseDataset
    ) -> dict[str, float]:
        """Evaluate model performance on a dataset.

        Args:
            embed_model: Model to generate sequence embeddings
            model: Model to evaluate predictions
            dataloader: DataLoader to evaluate on

        Returns:
            Dictionary mapping metric names to their computed values
        """
        # Configure evaluation
        model.eval()
        results = {}

        # Evaluate batches
        with torch.no_grad():
            for sequences, values in dataloader:
                # Generate embeddings and predictions
                embeddings = embed_model.embed_any_sequences(sequences)
                predictions = model(embeddings)

                # Update metrics
                values = values.to(self.device)
                for metric in self.metrics.values():
                    metric.update(predictions, values)

            # Compute final metrics
            for name, metric in self.metrics.items():
                results[name] = metric.compute().item()
                metric.reset()

        return results
