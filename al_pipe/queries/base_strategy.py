"""Abstract base class for AL query strategies."""

from abc import ABC, abstractmethod

from al_pipe.data_loader import base_data_loader

'''
class SelectionMethod:
    """
    Abstract base class for selection methods,
    which allow to select a subset of indices from the pool set as the next batch to label for Batch Active Learning.
    """
    def __init__(self):
        super().__init__()
        self.status = None  # can be used to report errors during selection

    def select(self, batch_size: int) -> torch.Tensor:
        """
        Select batch_size elements from the pool set
        (which is assumed to be given in the constructor of the corresponding subclass).
        This method needs to be implemented by subclasses.
        It is assumed that this method is only called once per object, since it can modify the state of the object.
        :param batch_size: Number of elements to select from the pool set.
        :return: Returns a torch.Tensor of integer type that contains the selected indices.
        """
        raise NotImplementedError()

    def get_status(self) -> Optional:
        """
        :return: Returns an object representing the status of the selection. If all went well, the method returns None.
        Otherwise, it might return a string or something different representing an error that occured.
        This is mainly useful for analyzing a lot of experiment runs.
        """
        return self.status
'''


class BaseQueryStrategy(ABC):
    """
    Abstract base class defining interface for active learning query strategies.

    Query strategies determine which samples from the unlabeled pool should be
    selected for labeling in each active learning iteration.
    """

    def __init__(self, selection_size: int) -> None:
        self.selection_size = selection_size

    @abstractmethod
    def select_samples(pool_loader: base_data_loader) -> None:
        """
        Select samples from unlabeled pool for labeling.

        Args:
            pool_loader: Pool of unlabeled samples to select from

        Returns:
            List of indices of selected samples from unlabeled pool
        """
        raise NotImplementedError()
