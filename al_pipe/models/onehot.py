"""Onehot encoding class for data embedding."""

import numpy as np
from util.data import Data
from util.general import avail_device


class OneHot:
    """
    A class for one-hot encoding categorical data.

    This class provides functionality to convert categorical data into one-hot encoded
    numerical representations, which can be used as input features for machine learning models.

    Attributes:
        params: Parameters for the one-hot encoding process
        device (str): Device to run computations on ('cuda' or 'cpu')
        al_data (Data): Data object containing the dataset to be encoded
    """

    def __init__(self, params, al_data: Data, device="cuda") -> None:  # noqa: D107
        self.params = params
        self.device = avail_device(device)

        self.al_data = al_data

    def get_embeddings(self, al_data: Data = None) -> np.ndarray:
        """
        Generate one-hot encoded embeddings for a specified column in the data.

        Args:
            al_data (Data, optional): The data object containing a pandas DataFrame.
                                       If not provided, self.al_data is used.
            TODO: write test for it

        Returns:
            np.ndarray: A 2D NumPy array with one-hot encoded data.
        """
        # Use provided al_data or fall back to self.al_data
        if al_data is None:
            al_data = self.al_data

        # Could be more versitile for protein data as well
        # TODO: need to add column_name to the dataset
        # values = df[column_name]
        values = 0

        # Determine the unique categories and create a mapping.
        categories = np.unique(values)
        mapping = {cat: idx for idx, cat in enumerate(categories)}

        # Map the values to indices.
        # Here we assume that the column is a pandas Series, so we can use .map()
        indices = values.map(mapping).to_numpy()

        # Create a one-hot matrix: each row is the identity vector for that category.
        one_hot_matrix = np.eye(len(categories), dtype=np.float32)

        # Use the indices to pick the one-hot encoded rows.
        embeddings = one_hot_matrix[indices]

        return embeddings
