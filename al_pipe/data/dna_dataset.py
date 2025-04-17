"""Specialized dataset class(es) for DNA data."""

from copy import deepcopy

import pandas as pd
import torch

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder
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
        embedding_model: BaseStaticEmbedder,
    ) -> None:
        super().__init__(data_path, data_name, batch_size, train_val_test_pool_split, max_length, embedding_model)
        self.data: pd.DataFrame = self._load_data()
        self.max_length: int = max_length

        if self.embedding_model is not None:
            self.update_embedded_data()

    # TODO: this is such shit design (you need to update embedded data every run)
    def update_embedded_data(self) -> None:
        """Update the embedded data."""
        self.embedded_data: list[torch.Tensor] = self._embed_data()

    def return_subset(self, indices: list[int]) -> "DNADataset":
        """Return a subset of the data."""
        new_dataset = deepcopy(self)
        new_dataset.data = self.data.iloc[indices]
        new_dataset.update_embedded_data()
        return new_dataset

    def get_subset(self, indices: list[int]) -> pd.DataFrame:
        """Get a subset of the data."""
        return self.data.iloc[indices]

    def delete(self, indices: list[int]) -> None:
        """Delete the items at the given indices."""
        for index in sorted(indices, reverse=True):
            del self.data[index]

    def append(self, data: pd.DataFrame) -> None:
        """Append the given data to the data."""
        self.data = pd.concat([self.data, data], ignore_index=True)

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
