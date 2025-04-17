"""Abstract base class for dataset."""

import os

from abc import ABC, abstractmethod

import torch

from torch.nn import functional as F
from torch.utils.data import Dataset

from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder


class BaseDataset(Dataset, ABC):
    """Abstract base class for dataset."""

    def __init__(
        self,
        data_path: str,
        data_name: str,
        batch_size: int,
        train_val_test_pool_split: list[float],
        max_length: int,
        embedding_model: BaseStaticEmbedder | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_path: str, the path to the data
            data_name: str, the name of the data
            batch_size: int, the batch size
            train_val_test_pool_split: list[float], the split of the data
            embedding_model: BaseStaticEmbedder, the embedding model
        """
        self.data_path = os.path.join(data_path, data_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_val_test_pool_split = train_val_test_pool_split
        print(f"In BaseDataset: Type of embedding_model: {type(embedding_model)}")
        print(f"In BaseDataset: embedding_model.__class__.__module__: {embedding_model.__class__.__module__}")
        self.embedding_model = embedding_model

    @abstractmethod
    def _load_data(self) -> list[torch.Tensor]:
        """Load data from the data path."""
        raise NotImplementedError()

    @abstractmethod
    def _embed_data(self) -> list[torch.Tensor]:
        """Embed the data."""
        raise NotImplementedError()

    def delete(self, indices: list[int]) -> None:
        """Delete the items at the given indices."""
        for index in sorted(indices, reverse=True):
            del self.data[index]

    def dna_collate_fn(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]], max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for DNA sequences of various lengths.
        """
        inputs = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])

        processed_inputs = []
        for tensor in inputs:
            if tensor.shape[0] < max_length:
                processed_inputs.append(F.pad(tensor, (0, 0, 0, max_length - tensor.shape[0]), "constant", 0))
                # padding = torch.zeros(max_length - tensor.shape[0], *tensor.shape[1:])
                # processed_inputs.append(torch.cat([tensor, padding], dim=0))
            else:
                processed_inputs.append(tensor[:max_length])
        return torch.stack(processed_inputs), labels
