"""Specialized dataset class(es) for DNA data."""

import pandas as pd

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.util.data import load_data


class DNADataset(BaseDataset):
    """Dataset for DNA data."""

    def __init__(
        self,
        data_path: str,
        data_name: str,
        batch_size: int,
        train_val_test_pool_split: list[float],
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__(data_path, data_name, batch_size, train_val_test_pool_split, num_workers, pin_memory)
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load data from the data path.

        This method utilizes the load_data function to read the dataset from the specified
        data path. It returns the loaded data as a list of torch tensors.

        Returns:
            list[torch.Tensor]: A list of tensors containing the loaded DNA sequences and their values.
        """
        return load_data(self.data_path)

    def get_sequences(self) -> pd.Series:
        """Get the sequences from the dataset.

        Returns:
            pd.Series: A pandas Series containing the sequences from the dataset.
        """
        return self.data["sequences"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, float]:
        return self.data["sequences"].iloc[index], self.data[index]["values"].iloc[index]
