#!/usr/bin/env python3
"""
main.py: Orchestrates the full active learning pipeline for DNA embedding.
"""

import os

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from al_pipe.data.dna_dataset import DNADataset
from al_pipe.data_loader.dna_data_loader import DNADataLoader
from al_pipe.util.general import (
    avail_device,
    seed_all,
)

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

    # ==========================
    # 1. Set up Device and Seed
    # ==========================
    device = avail_device(cfg.device)
    print(f"Device: {device}")
    seed_all(cfg.seed)  # A helper that sets seed for torch, numpy, etc.

    # ==========================
    # 2. Instantiate the Static Embedding Model
    # ==========================

    regressor = hydra.utils.instantiate(cfg.regression)

    print(regressor)
    print(f"Type of regressor: {type(regressor)}")
    print(regressor.__class__.__module__)
    # ==========================
    # 3. Prepare the Dataset and DataLoader
    # ==========================
    embedding_model = hydra.utils.instantiate(cfg.model)
    print(embedding_model)
    print(f"Type of embedding_model: {type(embedding_model)}")
    print(embedding_model.__class__.__module__)

    dataset = DNADataset(
        data_path=os.path.join(cfg.paths.data_dir, cfg.datasets.data_path),
        data_name=cfg.datasets.data_name,
        batch_size=cfg.datasets.batch_size,
        train_val_test_pool_split=cfg.datasets.train_val_test_pool_split,
        max_length=cfg.datasets.MAX_LENGTH,
        embedding_model=embedding_model,
    )

    full_data_loader = DNADataLoader(
        dataset=dataset,
        batch_size=cfg.datasets.batch_size,
        num_workers=cfg.datasets.num_workers,
        pin_memory=cfg.datasets.pin_memory,
        shuffle=cfg.datasets.shuffle,
        first_batch_strategy=hydra.utils.instantiate(cfg.first_batch),
    )

    # ==========================
    # 4. Set Up Active Learning Components
    # ==========================
    query_strategy = hydra.utils.instantiate(cfg.query)
    print(query_strategy)

    # ==========================
    # 5. Instantiate Trainer and logger
    # ==========================
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    print(trainer)

    # ==========================
    # 6. Active Learning Loop
    # ==========================
    n_iterations = cfg.active_learning.al_iterations

    for iteration in range(n_iterations):
        print(f"\n=== Active Learning Iteration {iteration + 1}/{n_iterations} ===")

        trainer.fit(
            regressor,
            train_dataloaders=full_data_loader.get_train_loader(),
            val_dataloaders=full_data_loader.get_val_loader(),
        )

        query_strategy.select_samples(full_data_loader)

    # ==========================
    # 7. Final Evaluation
    # ==========================


if __name__ == "__main__":
    main()
