#!/usr/bin/env python3
"""
main.py: Orchestrates the full active learning pipeline for DNA embedding.
"""

import os

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from al_pipe.data.dna_dataset import DNADataset
from al_pipe.labeling.in_silico_labeler import InSilicoLabeler
from al_pipe.training.trainer import Trainer
from al_pipe.util.general import (
    avail_device,
    initialize_first_batch_strategy,
    initialize_model,
    initialize_query_strategy,
    seed_all,
)

# Import data handling and modules

# Import models – here we assume model types are embedding models

# Import first batch and query strategies

# Import labeling module (the oracle/simulation)

# Import trainer and evaluator to run training loop and assess performance

# Import common utility functions


load_dotenv()  # This loads environment variables from the .env file


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function to orchestrate the active learning pipeline for DNA embedding.

    This function performs the following steps:
    1. Sets up the device (CPU or GPU) and random seed for reproducibility.
    2. Prepares the dataset based on the configuration.
    3. Instantiates the embedding model.`
    4. Sets up active learning components including first-batch strategy, query strategy, and labeling module.
    5. Initializes the trainer and evaluator.
    6. Runs the active learning loop for a specified number of iterations.
    7. Evaluates the model on the entire dataset at the end.

    Args:
        cfg: Hydra configuration object containing all parameters

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the dataset or model configuration is invalid.
    """
    print(OmegaConf.to_yaml(cfg))
    # print(f"data path: {cfg.paths.data_dir}")
    # config_output_path = "./al_pipe/configs/hydra_example.yaml"
    # with open("hydra_example.yaml", "w") as f:
    #     OmegaConf.save(config=cfg, f=f)
    # print(f"Configuration saved to {config_output_path}")

    # ==========================
    # 1. Set up Device and Seed
    # ==========================
    device = avail_device(cfg.device)
    print(f"Device: {device}")
    seed_all(cfg.seed)  # A helper that sets seed for torch, numpy, etc.

    # # ==========================
    # # 2. Prepare the Dataset
    # # ==========================
    # # This DNADataset should be implemented to load DNA sequences, possibly from a CSV/fasta etc.
    dataset = DNADataset(os.path.join(cfg.paths.data_dir, cfg.datasets.data_path), cfg.datasets.data_name)
    # os.path.join(cfg.paths.data_dir, cfg.datasets.data_path, cfg.datasets.data_name),
    # **(cfg.datasets.params or {}) #If more params are needed

    # # ==========================
    # # 3. Instantiate the Static Embedding Model
    # # ==========================
    # TODO: what is the type of cfg.model?
    model = initialize_model(cfg.model, dataset)
    print(model)

    # # ==========================
    # # 4. Set Up Active Learning Components
    # # ==========================
    # First-Batch Strategy: to select an initial batch
    first_batch_strategy = initialize_first_batch_strategy(cfg.first_batch, dataset)
    print(first_batch_strategy)
    # Query Strategy: for iterative selection of samples
    query_strategy = initialize_query_strategy(cfg.query, dataset)
    print(query_strategy)

    # Labeling Module: simulation of an oracle to provide labels
    labeler = InSilicoLabeler(cfg.labeling.path, cfg.labeling.data_name)
    print(labeler)

    # # ==========================
    # # 5. Instantiate Trainer and Evaluator
    # # ==========================
    # trainer = Trainer(model, device, **(cfg.trainer or {}))

    # TODO: callbacks and logger
    # log.info("Instantiating callbacks...")
    # callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # # log.info("Instantiating loggers...")
    # logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    print(trainer)
    # evaluator = Evaluator(**(cfg.evaluation or {}))

    # # ==========================
    # # 6. Active Learning Loop
    # # ==========================
    # # Initially, use the first-batch strategy (or a default random split) to select the starting labeled set.
    # if first_batch_strategy is not None:
    #     labeled_idxs, unlabeled_idxs = first_batch_strategy.select_initial_samples(
    #         dataset, cfg.active_learning.initial_batch_size
    #     )
    # else:
    #     # Default: randomly select a fixed number for the initial training set.
    #     total_samples = len(dataset)
    #     labeled_idxs = torch.randperm(total_samples)[: cfg.active_learning.initial_batch_size].tolist()
    #     unlabeled_idxs = [i for i in range(total_samples) if i not in labeled_idxs]

    # # Convert indices to dataset splits (this assumes your dataset supports indexing)
    # labeled_data = dataset.get_subset(labeled_idxs)
    # unlabeled_data = dataset.get_subset(unlabeled_idxs)

    # # Run the iterative Active Learning loop for a fixed number of iterations
    # n_iterations = cfg.active_learning.iterations
    # acquisition_batch_size = cfg.active_learning.acquisition_batch_size

    # for iteration in range(n_iterations):
    #     print(f"\n=== Active Learning Iteration {iteration + 1}/{n_iterations} ===")

    #     # Train the model on the current labeled data
    #     trainer.train(labeled_data)

    #     # Evaluate on unlabeled pool and/or validation set if available
    #     metrics = evaluator.evaluate(model, labeled_data)
    #     print(f"Evaluation metrics: {metrics}")

    #     # Use the query strategy to select new samples from unlabeled_data
    #     queried_idxs = query_strategy.select_samples(model, unlabeled_data, batch_size=acquisition_batch_size)

    #     # Query the labeling module (simulated oracle) to obtain ground truth labels for selected samples
    #     new_labeled_data = labeler.label(unlabeled_data, queried_idxs)

    #     # Update the labeled set and remove newly labeled indices from the unlabeled pool
    #     labeled_data.add(new_labeled_data)
    #     unlabeled_data.remove(queried_idxs)

    #     # Optionally, save checkpoints, log results, or adjust hyperparameters here

    # # ==========================
    # # 7. Final Evaluation
    # # ==========================
    # final_metrics = evaluator.evaluate(model, dataset)
    # print("Final evaluation metrics:", final_metrics)


if __name__ == "__main__":
    main()
