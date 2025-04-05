"""Main Class."""

from util.general import seed_all
from al_pipe.util.data import Data


class ActivePipe:  # noqa: D101
    def __init__(self, path, dataset_name, exp_name="ActiveLearning", seed=1, num_cpus=4, device="cuda") -> None:  # noqa: D417
        """
        Initialize the ActivePipe class for active learning experiments.

        Args:
            exp_name (str, optional): Name of the experiment. Defaults to 'ActiveLearning'.
            seed (int, optional): Random seed for numpy. Defaults to 1.
            num_cpus (int, optional): Number of CPU cores to use. Defaults to 4.
            device (str, optional): Device to run on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """  # noqa: E501
        # Setting vars
        self.device = device
        self.exp_name = exp_name
        self.path = path
        self.dataset_name = dataset_name

        # Seeding random states
        seed_all(seed)

    def initialize_data(self):
        """Initalizing data with Data object."""
        self.dataset = Data(path=self.path, dataset_name=self.dataset_name, seed=self.seed)
        self.test_data = self.dataset.get_test_data()

    def initialize_model(self):
        """Initialize model."""

    def initialize_active_learning_strategy(self):
        """Initialize ALC."""
        pass
