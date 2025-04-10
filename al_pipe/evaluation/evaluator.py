"""Evaluator module for assessing model performance during active learning.

This module provides functionality to evaluate embedding models during the active learning
process, tracking various metrics like loss, accuracy, and other relevant performance indicators.
"""

import torch

from torch.utils.data import DataLoader

from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder
from al_pipe.util.general import avail_device


class Evaluator:
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
        self.metrics = metrics or {"mse": torch.nn.MSELoss()}
        self.batch_size = batch_size
        self.device = avail_device(device)

    def evaluate(self, model: BaseStaticEmbedder, dataset: any) -> dict[str, float]:
        """Evaluate the model on the given dataset.

        Args:
            model: The embedding model to evaluate.
            dataset: Dataset to evaluate on.

        Returns:
            Dictionary containing computed metrics.
        """
        model.eval()  # Set model to evaluation mode
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        results = {}
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                total_metric = 0.0
                num_batches = 0

                for batch in dataloader:
                    # Assuming batch contains sequences and values
                    sequences, values = batch
                    embeddings = model.embed_any_sequences(sequences)

                    # Compute metric
                    metric_value = metric_fn(embeddings, values.to(self.device))
                    total_metric += metric_value.item()
                    num_batches += 1

                results[metric_name] = total_metric / num_batches

        return results
