"""Main Class."""

import numpy as np
import torch

# from al_pipe.util.data import Data


class ActivePipe:  # noqa: D101
    def __init__(self, exp_name="ActiveLearning", seed=1, run=1, num_cpus=4, device="cuda") -> None:
        """
        Initialize the ActivePipe class for active learning experiments.

        Args:
            exp_name (str, optional): Name of the experiment. Defaults to 'ActiveLearning'.
            seed (int, optional): Random seed for numpy. Defaults to 1.
            run (int, optional): Random seed for PyTorch. Defaults to 1.
            num_cpus (int, optional): Number of CPU cores to use. Defaults to 4.
            device (str, optional): Device to run on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """  # noqa: E501
        self.device = device
        np.random.seed(seed)
        torch.manual_seed(run)
        torch.backends.cudnn.enabled = False
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(self.device if use_cuda else "cpu")
        print("device: ", self.device)

    def initialize_data(self):
        """Initalizing data."""
        # self.dataset = Data(path, dataset_name, batch_size, adata, test_fraction, self.seed, custom_test)
        # self.test_data = self.dataset.get_test_data()
        # self.dataset_name = dataset_name
        # self.path = path
        pass

    def initialize_model(self):
        """Initialize model."""
        pass

    def initialize_active_learning_strategy(self):
        """Initialize ALC."""
        pass
