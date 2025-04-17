"""
util.init is a module that initializes the model, first batch strategy, query strategy, and evaluator.
"""

from al_pipe.data.base_dataset import BaseDataset
from al_pipe.embedding_models.static.base_static_embedder import BaseStaticEmbedder
from al_pipe.embedding_models.static.onehot_embedding import OneHotEmbedder
from al_pipe.evaluation.evaluator import Evaluator
from al_pipe.queries.base_strategy import BaseQueryStrategy
from al_pipe.queries.random_sampling import RandomQueryStrategy


def initialize_model(model_config: dict) -> BaseStaticEmbedder:
    """Initialize a model based on the model configuration."""
    if model_config["name"] == "OneHotEmbedder" and model_config["type"] == "static":
        return OneHotEmbedder(model_config["device"])
    else:
        raise ValueError(f"Model class {model_config['type']} must inherit from BaseStaticEmbedder")


def initialize_query_strategy(query_config: dict, dataset: BaseDataset) -> BaseQueryStrategy:
    """Initialize a query strategy based on the query configuration."""
    if query_config["type"] == "RandomQueryStrategy":
        return RandomQueryStrategy(dataset, query_config["batch_size"])
    else:
        raise ValueError(f"Query strategy class {query_config['type']} is not supported.")


def initialize_evaluator(batch_size: int, device: str):
    """Initialize an evaluator based on the model and device."""
    return Evaluator(batch_size=batch_size, device=device)
