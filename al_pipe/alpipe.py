"""Main Class."""

from util.general import seed_all
from al_pipe.util.data import Data


class ActivePipe:
    """Class ActivePipe brings together different components of the Active learning pipeline."""

    def __init__(
        self,
        path: str,
        dataset_name: str,
        exp_name: str = "ActiveLearning",
        embedding_mode: str = "onehot",
        seed: int = 1,
        num_cpus: int = 4,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the ActivePipe class for active learning experiments.

        Args:
            path (str): Path to the dataset directory
            dataset_name (str): Name of the dataset file
            exp_name (str, optional): Name of the experiment. Defaults to 'ActiveLearning'.
            embedding_mode (str, optional): Mode for embedding the data. Defaults to 'onehot'.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            num_cpus (int, optional): Number of CPU cores to use. Defaults to 4.
            device (str, optional): Device to run on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """  # noqa: E501
        # Setting vars
        self.device = device
        self.exp_name = exp_name
        self.path = path
        self.dataset_name = dataset_name
        self.embedding_mode = embedding_mode

        # Seeding random states
        seed_all(seed)

    def initialize_data(self):
        """Initalizing data with Data object."""
        self.dataset = Data(
            path=self.path, dataset_name=self.dataset_name, embedding_mode=self.embedding_mode, seed=self.seed
        )

        # self.test_data = self.dataset.get_test_data()

    def initialize_model(self):
        """Initialize model."""

    def initialize_active_learning_strategy(self):
        """Initialize ALC."""
        pass

    def train_start(self):
        """Kick starting the whole active learning pipeline."""
        pass
