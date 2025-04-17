"""Specialized dataset class(es) for DNA data."""

import pandas as pd
import torch

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder
from al_pipe.first_batch.base_first_batch import FirstBatchStrategyFactory
from al_pipe.util.data import load_data


class DNADataset(BaseDataset):
    """Dataset for DNA data."""

    def __init__(
        self,
        data_path: str,
        data_name: str,
        batch_size: int,
        train_val_test_pool_split: list[float],
        max_length: int,
        embedding_model: BaseStaticEmbedder | None = None,
        first_batch_strategy: FirstBatchStrategyFactory | None = None,
    ) -> None:
        super().__init__(
            data_path, data_name, batch_size, train_val_test_pool_split, embedding_model, first_batch_strategy
        )
        self.data: pd.DataFrame = self._load_data()
        self.max_length: int = max_length

        if embedding_model is not None:
            self.embedded_data: list[torch.Tensor] = self._embed_data()
        if first_batch_strategy is not None:
            self.first_batch_strategy = first_batch_strategy

    def _embed_data(self) -> list[torch.Tensor]:
        """Embed the data."""
        return self.embedding_model.embed_any_sequences(self.data["sequences"])

    def _load_data(self) -> pd.DataFrame:
        """Load data from the data path.

        This method utilizes the load_data function to read the dataset from the specified
        data path. It returns the loaded data as a list of torch tensors.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the loaded DNA sequences and their values.
        """
        return load_data(self.data_path)

    def return_label(self, sequences: list[str]) -> list[float]:
        """Return the labels for the given sequences.

        Args:
            sequences (list[str]): A list of DNA sequences.

        Returns:
            list[float]: A list of labels corresponding to the given sequences.
        """
        labels = []
        for sequence in sequences:
            if sequence in self.data["sequences"].values:
                labels.append(self.data.loc[self.data["sequences"] == sequence, "values"].values[0])
            else:
                raise ValueError(f"Sequence {sequence} not found in the dataset.")
        return labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embedded_data[index].float(), torch.tensor(self.data["values"].iloc[index]).float()
