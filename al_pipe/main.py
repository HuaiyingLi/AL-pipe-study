#!/usr/bin/env python3
"""
main.py: Orchestrates the full active learning pipeline for DNA embedding.
"""

import argparse

import torch
import yaml

# Import data handling and modules
from al_pipe.data.base_dataset import BaseDataset  # dataset class

# Import models – here we assume model types are embedding models
from al_pipe.embedding_models.static.base_DNA_embedder import BaseDNAEmbedder
from al_pipe.evaluation.evaluator import Evaluator

# Import first batch and query strategies
from al_pipe.first_batch.base_first_batch import FirstBatchStrategy

# Import labeling module (the oracle/simulation)
from al_pipe.labeling.in_silico_labeler import InSilicoLabeler

# Import trainer and evaluator to run training loop and assess performance
from al_pipe.queries.base_strategy import BaseQueryStrategy
from al_pipe.training.trainer import Trainer

# Import common utility functions
from al_pipe.util.general import avail_device, seed_all


def main() -> None:
    """Main function to orchestrate the active learning pipeline for DNA embedding.

    This function performs the following steps:
    1. Parses the configuration from a YAML file.
    2. Sets up the device (CPU or GPU) and random seed for reproducibility.
    3. Prepares the dataset based on the configuration.
    4. Instantiates the embedding model.
    5. Sets up active learning components including first-batch strategy, query strategy, and labeling module.
    6. Initializes the trainer and evaluator.
    7. Runs the active learning loop for a specified number of iterations.
    8. Evaluates the model on the entire dataset at the end.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the dataset or model configuration is invalid.
    """
    # ==========================
    # 1. Parse Configuration
    # ==========================
    parser = argparse.ArgumentParser(description="Active Learning Pipeline for DNA Embedding")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load configuration settings from YAML file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ==========================
    # 2. Set up Device and Seed
    # ==========================
    device = avail_device(config.get("device", "cuda"))
    seed_all(args.seed)  # A helper that sets seed for torch, numpy, etc.

    # ==========================
    # 3. Prepare the Dataset
    # ==========================
    # Assume dataset configuration specifies a file location and parameters needed by your dataset class.
    dataset_config = config["dataset"]
    # This BaseDataset should be implemented to load DNA sequences, possibly from a CSV/fasta etc.
    dataset = BaseDataset(dataset_config["data_path"], **dataset_config.get("params", {}))

    # ==========================
    # 5. Instantiate the Embedding Model
    # ==========================
    model_config = config["model"]
    # Use globals() or a factory to select the proper model
    model_cls = globals()[model_config["type"]]
    model: BaseDNAEmbedder = model_cls(sequence_data=dataset.sequences, **model_config.get("params", {}))
    model.to(device)

    # ==========================
    # 6. Set Up Active Learning Components
    # ==========================
    # First-Batch Strategy: to select an initial batch
    first_batch_config = config.get("first_batch", {})
    if first_batch_config:
        fb_cls = globals()[first_batch_config["type"]]
        first_batch_strategy: FirstBatchStrategy = fb_cls(**first_batch_config.get("params", {}))
    else:
        first_batch_strategy = None

    # Query Strategy: for iterative selection of samples
    query_config = config["query_strategy"]
    qs_cls = globals()[query_config["type"]]
    query_strategy: BaseQueryStrategy = qs_cls(**query_config.get("params", {}))

    # Labeling Module: simulation of an oracle to provide labels
    labeling_config = config["labeling"]
    labeler = InSilicoLabeler(**labeling_config.get("params", {}))

    # ==========================
    # 7. Instantiate Trainer and Evaluator
    # ==========================
    trainer = Trainer(model, device, **config.get("trainer", {}))
    evaluator = Evaluator(**config.get("evaluation", {}))

    # ==========================
    # 8. Active Learning Loop
    # ==========================
    # Initially, use the first-batch strategy (or a default random split) to select the starting labeled set.
    if first_batch_strategy is not None:
        labeled_idxs, unlabeled_idxs = first_batch_strategy.select_initial_samples(
            dataset, config["active_learning"]["initial_batch_size"]
        )
    else:
        # Default: randomly select a fixed number for the initial training set.
        total_samples = len(dataset)
        labeled_idxs = torch.randperm(total_samples)[: config["active_learning"]["initial_batch_size"]].tolist()
        unlabeled_idxs = [i for i in range(total_samples) if i not in labeled_idxs]

    # Convert indices to dataset splits (this assumes your dataset supports indexing)
    labeled_data = dataset.get_subset(labeled_idxs)
    unlabeled_data = dataset.get_subset(unlabeled_idxs)

    # Run the iterative Active Learning loop for a fixed number of iterations
    n_iterations = config["active_learning"]["iterations"]
    acquisition_batch_size = config["active_learning"]["acquisition_batch_size"]

    for iteration in range(n_iterations):
        print(f"\n=== Active Learning Iteration {iteration + 1}/{n_iterations} ===")

        # Train the model on the current labeled data
        trainer.train(labeled_data)

        # Evaluate on unlabeled pool and/or validation set if available
        metrics = evaluator.evaluate(model, labeled_data)
        print(f"Evaluation metrics: {metrics}")

        # Use the query strategy to select new samples from unlabeled_data
        queried_idxs = query_strategy.select_samples(model, unlabeled_data, batch_size=acquisition_batch_size)

        # Query the labeling module (simulated oracle) to obtain ground truth labels for selected samples
        new_labeled_data = labeler.label(unlabeled_data, queried_idxs)

        # Update the labeled set and remove newly labeled indices from the unlabeled pool
        labeled_data.add(new_labeled_data)
        unlabeled_data.remove(queried_idxs)

        # Optionally, save checkpoints, log results, or adjust hyperparameters here

    # ==========================
    # 9. Final Evaluation
    # ==========================
    final_metrics = evaluator.evaluate(model, dataset)
    print("Final evaluation metrics:", final_metrics)


if __name__ == "__main__":
    main()
