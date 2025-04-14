"""
util.init is a module that initializes the model, first batch strategy, query strategy, and evaluator.
"""

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder
from al_pipe.embedding_models.static.onehot_embedding import OneHotEmbedder
from al_pipe.evaluation.evaluator import Evaluator
from al_pipe.first_batch.base_first_batch import FirstBatchStrategy
from al_pipe.first_batch.random import RandomFirstBatch
from al_pipe.queries.base_strategy import BaseQueryStrategy
from al_pipe.queries.random_sampling import RandomQueryStrategy


def initialize_model(model_config: dict, dataset: BaseDataset) -> BaseStaticEmbedder:
    """Initialize a model based on the model configuration."""
    if model_config["name"] == "OneHotEmbedder" and model_config["type"] == "static":
        return OneHotEmbedder(dataset, model_config["device"])
    else:
        raise ValueError(f"Model class {model_config['type']} must inherit from BaseStaticEmbedder")


def initialize_first_batch_strategy(first_batch_config: dict, dataset: BaseDataset) -> FirstBatchStrategy:
    """Initialize a first batch strategy based on the first batch configuration."""
    if first_batch_config["type"] == "RandomFirstBatch":
        return RandomFirstBatch(dataset, first_batch_config["batch_size"])
    else:
        raise ValueError(f"First batch strategy class {first_batch_config['type']} is not supported.")


def initialize_query_strategy(query_config: dict, dataset: BaseDataset) -> BaseQueryStrategy:
    """Initialize a query strategy based on the query configuration."""
    if query_config["type"] == "RandomQueryStrategy":
        return RandomQueryStrategy(dataset, query_config["batch_size"])
    else:
        raise ValueError(f"Query strategy class {query_config['type']} is not supported.")


def initialize_evaluator(batch_size: int, device: str):
    """Initialize an evaluator based on the model and device."""
    return Evaluator(batch_size=batch_size, device=device)
