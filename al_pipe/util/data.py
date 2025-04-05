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
from .general import seed_all


class Data:
    """
    A class to handle data loading and splitting for active learning.

    This class manages the loading of data from disk and splitting it into labeled
    and unlabeled sets for active learning purposes.

    Attributes:
        path (str): Path to the data file
        data_name (str): Name of the dataset
        batch_size (int): Size of batches for active learning
        test_fraction (float): Fraction of data to use for testing (default: 0.1)
        seed (int): Random seed for reproducibility (default: 1)
        custom_test (str, optional): Path to custom test set

    """  # noqa: E501

    def __init__(self, path, data_name, batch_size=32, test_fraction=0.1, seed=42) -> None:  # noqa: D107, E501
        # get the input file
        self.data_path = os.path.join(path, data_name)
        self.batch_size = batch_size

        # seed the program
        seeded_state = seed_all(seed)

        # Load the CSV data into a DataFrame
        data = self.load_data(self.data_path)

        # Split off the test set (e.g., 20% of the data)
        data_remaining, self.test = train_test_split(data, test_size=0.2, random_state=seeded_state)

        # 2. From the remaining data, split out the initial labeled data and the pool.  # noqa: E501
        # allocate 20% of the remaining data for an initial labeled set  # noqa: E501
        # (which we can later further split into training and validation) and 80% as the pool.  # noqa: E501
        initial_labeled, self.pool = train_test_split(data_remaining, test_size=0.8, random_state=seeded_state)  # noqa: E501

        # 3. Optionally, split the initial labeled set into a training set and a validation set.  # noqa: E501
        # For example, 80% training and 20% validation:
        self.train, self.val = train_test_split(initial_labeled, test_size=0.2, random_state=seeded_state)  # noqa: E501

    def get_test_data(self) -> None:
        """Returning test data."""
        return self.test

    def get_train_data(self) -> None:
        """Returning train data."""
        return self.train

    def get_pool_data(self) -> None:
        """Returning pool data."""
        return self.pool

    def get_val_data(self) -> None:
        """Returning val data."""
        return self.val

    def load_data(self, data_path: str) -> pd.DataFrame:
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
        else:
            raise ValueError("Unsupported file type. Only CSV are supported.")

        return data
