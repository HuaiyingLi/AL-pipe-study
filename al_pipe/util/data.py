"""
    1. Load data from disk
    2. Splits into L and U.

------
Write function to load data

Split data into labeled/unlabeled

Test the function on a small sample

"""  # noqa: D205

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from al_pipe.util.general import avail_device, seed_all
from torch.utils.data import Dataset

from al_pipe.embedding_models.static.onehot_embedding import OneHotEmbedding
# TODO: add isort input sort basically


def load_data(data_path: str) -> pd.DataFrame:
    # load data from path
    """
    Load data from the given file path.

    Parameters:
    - data_path: str, the path to the data file (e.g., CSV)

    Returns:
    - pd.DataFrame containing the loaded data.
    TODO: could support different file type
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    if data_path.lower().endswith(".csv"):
        # default separator here is \t
        data = pd.read_csv(data_path, sep="\t")
        if data.shape[1] < 2:
            raise ValueError("CSV file must contain at least two columns.")
        data.columns = ["sequences", "values"]
    else:
        raise ValueError("Unsupported file type. Only CSV are supported.")

    return data


class CustomSequenceDataset(Dataset):
    """CustomSequenceDataset here created to fit different embedding needs."""

    def __init__(
        self, data: pd.DataFrame, embedding_mode: str = "onehot", vocab_type: str = "DNA", device: str = "cuda"
    ) -> None:
        """
        Initialize the CustomSequenceDataset.

        Args:
            data (pd.DataFrame): DataFrame containing sequences and their values.
            embedding_mode (str, optional): Type of embedding to use. Defaults to "onehot".
            vocab_type (str, optional): Type of sequence vocabulary. Defaults to "DNA".
            device (str, optional): Device to run computations on. Defaults to "cuda".

        The dataset expects a DataFrame with 'sequences' and 'values' columns. The sequences
        will be embedded according to the specified embedding_mode, and the values will be
        converted to tensors. Currently supports DNA sequences (vocab_type='DNA') with
        onehot encoding.
        """
        super().__init__()
        # torch.tensor(self.sequence_values["values"].to_numpy())
        self.device = avail_device(device)
        self.sequences_val_pair = data
        self.sequences = self.sequences_val_pair["sequences"]
        self.values = torch.tensor(self.sequences_val_pair["values"].to_numpy())
        # TODO: account for different input later
        self.vocab_size = 4 if vocab_type == "DNA" else 20  # 4 for DNA nucleotides, 20 for amino acids

        if embedding_mode == "onehot":
            embedding_model = OneHotEmbedding(self.sequences)
            self.embeded_sequences = embedding_model.embed_loaded_sequences()  # return pd.Series
        else:
            self.embeded_sequences = self.sequences

    def __len__(self) -> int:
        return len(self.embeded_sequences)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeded_sequences[idx], self.values[idx]


class Data:
    """
    A class to handle data loading and splitting for active learning.

    This class manages the loading of data from disk and splitting it into labeled
    and unlabeled sets for active learning purposes. It also handles data transformation
    if embeddings are needed for training and inference task.

    Attributes:
        path (str): Path to the data file
        data_name (str): Name of the dataset
        batch_size (int): Size of batches for active learning
        test_fraction (float): Fraction of data to use for testing (default: 0.1)
        seed (int): Random seed for reproducibility (default: 1)
        custom_test (str, optional): Path to custom test set

    """  # noqa: E501

    def __init__(
        self,
        path: str,
        data_name: str,
        batch_size: int = 32,
        embedding_mode: str = "onehot",
        test_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:  # noqa: D107, E501
        """Initializing the Data class."""
        # get the input file
        self.data_path = os.path.join(path, data_name)
        self.batch_size = batch_size
        self.data = {}
        self.datasets = {}
        self.data_splits = ["test", "train", "pool", "val"]
        # seed the program
        seeded_state = seed_all(seed)

        # Load the CSV data into a DataFrame
        data = load_data(self.data_path)

        # Split off the test set (e.g., 20% of the data)
        data_remaining, self.data["test"] = train_test_split(data, test_size=0.2, random_state=seeded_state)

        # 2. From the remaining data, split out the initial labeled data and the pool.  # noqa: E501
        # allocate 20% of the remaining data for an initial labeled set  # noqa: E501
        # (which we can later further split into training and validation) and 80% as the pool.  # noqa: E501
        initial_labeled, self.data["pool"] = train_test_split(data_remaining, test_size=0.8, random_state=seeded_state)  # noqa: E501

        # 3. Optionally, split the initial labeled set into a training set and a validation set.  # noqa: E501
        # For example, 80% training and 20% validation:
        self.data["train"], self.data["val"] = train_test_split(
            initial_labeled, test_size=0.2, random_state=seeded_state
        )  # noqa: E501

        self.datasets = {split: CustomSequenceDataset(self.data[split]) for split in self.data_splits}

    def get_datasets(self) -> CustomSequenceDataset:
        """Returning a set of datasets."""
        return self.datasets

    def get_test_data(self) -> CustomSequenceDataset:
        """Returning test data as a CustomSequenceDataset."""
        return self.datasets["test"]

    def get_train_data(self) -> CustomSequenceDataset:
        """Returning train data as a CustomSequenceDataset."""
        return self.dataset["train"]

    def get_pool_data(self) -> CustomSequenceDataset:
        """Returning pool data as a CustomSequenceDataset."""
        return self.dataset["pool"]

    def get_val_data(self) -> CustomSequenceDataset:
        """Returning val data as a CustomSequenceDataset."""
        return self.dataset["val"]
