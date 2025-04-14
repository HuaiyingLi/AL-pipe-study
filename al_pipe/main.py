#!/usr/bin/env python3
"""
main.py: Orchestrates the full active learning pipeline for DNA embedding.
"""

import os

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from al_pipe.data.dna_dataset import DNADataset
from al_pipe.data_loader.dna_data_loader import DnaDataLoader
from al_pipe.labeling.in_silico_labeler import InSilicoLabeler
from al_pipe.training.trainer import Trainer
from al_pipe.util.general import (
    avail_device,
    seed_all,
)
from al_pipe.util.init import (
    initialize_evaluator,
    initialize_first_batch_strategy,
    initialize_model,
    initialize_query_strategy,
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
    # # 2. Prepare the Dataset and DataLoader
    # # ==========================
    # # This DNADataset should be implemented to load DNA sequences, possibly from a CSV/fasta etc.
    dataset = DNADataset(
        os.path.join(cfg.paths.data_dir, cfg.datasets.data_path),
        cfg.datasets.data_name,
        batch_size=cfg.datasets.batch_size,
        train_val_test_pool_split=cfg.datasets.train_val_test_pool_split,
    )
    full_data_loader = DnaDataLoader(
        dataset,
        batch_size=cfg.datasets.batch_size,
        num_workers=cfg.datasets.num_workers,
        pin_memory=cfg.datasets.pin_memory,
        shuffle=cfg.datasets.shuffle,
    )
    # os.path.join(cfg.paths.data_dir, cfg.datasets.data_path, cfg.datasets.data_name),
    # **(cfg.datasets.params or {}) #If more params are needed

    # # ==========================
    # # 3. Instantiate the Static Embedding Model
    # # ==========================
    # TODO: what is the type of cfg.model?
    model = initialize_model(cfg.model, dataset)
    print(model)
    regressor = hydra.utils.instantiate(cfg.regression)
    print(regressor)
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
    labeler: InSilicoLabeler = InSilicoLabeler(cfg.labeling.path, cfg.labeling.data_name)
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
    # Initially, use the first-batch strategy (or a default random split) to select the starting labeled set.
    if first_batch_strategy is not None:
        # TODO: ZELUN THIS IS SHIT DESIGN NEED TO FIX
        # MOVE DATA_SIZE TO THE CONSTRUCTOR OF THE DATASET class
        # FIRST_BATCH STRATEGY SHOULD SET THE BATCH_SIZE NOT THE SPLIT SIZE
        first_batch_loader, test_batch_loader, val_batch_loader, pool_loader = first_batch_strategy.select_first_batch(
            data_loader=full_data_loader, data_size=cfg.datasets.train_val_test_pool_split
        )
    else:
        raise ValueError("First batch strategy is not set")

    # # Convert indices to dataset splits (this assumes your dataset supports indexing)
    # labeled_data = dataset.get_subset(labeled_idxs)
    # unlabeled_data = dataset.get_subset(unlabeled_idxs)

    evaluator = initialize_evaluator(model, device)
    # Run the iterative Active Learning loop for a fixed number of iterations
    n_iterations = cfg.active_learning.al_iterations
    acquisition_batch_size = cfg.active_learning.acquisition_batch_size

    train_loader = first_batch_loader
    for iteration in range(n_iterations):
        print(f"\n=== Active Learning Iteration {iteration + 1}/{n_iterations} ===")

        # Train the model on the current labeled data
        trainer.train(train_loader, test_batch_loader, val_batch_loader)

        # Evaluate on unlabeled pool and/or validation set if available
        metrics = evaluator.evaluate(model, regressor, dataset)
        print(f"Evaluation metrics: {metrics}")

        # Use the query strategy to select new samples from unlabeled_data
        queried_idxs = query_strategy.select_samples(model, pool_loader, batch_size=acquisition_batch_size)

        # Query the labeling module (simulated oracle) to obtain ground truth labels for selected samples
        new_labeled_data = labeler.label(pool_loader, queried_idxs)

        # Update the labeled set and remove newly labeled indices from the unlabeled pool
        # By appending, we essentially mean training on all the data
        train_loader.update_train_dataset(queried_idxs, new_labels=new_labeled_data, mode="append")

        pool_loader.update_pool_dataset(queried_idxs, mode="remove")

        # Optionally, save checkpoints, log results, or adjust hyperparameters here

    # ==========================
    # 7. Final Evaluation
    # ==========================
    final_metrics = evaluator.evaluate(model, dataset)
    print("Final evaluation metrics:", final_metrics)


if __name__ == "__main__":
    main()
