"""CLI entry point for running experiments."""

import argparse  # noqa: D100

from al_pipe.alpipe import ActivePipe

parser = argparse.ArgumentParser()
parser.add_argument("--fb_strategy", type=str, default="Random")
args = parser.parse_args()

fb_strategy = args.fb_strategy  # first batch strategy

interface = ActivePipe(weight_bias_track=True, exp_name=fb_strategy, device="cuda", seed=1)

# data_path
path = "./dataset/random_promo"
interface.initialize_data(path=path, dataset_name="sub_sample_pTpA_All", batch_size=32)

interface.initialize_model(epochs=20, hidden_size=64)
interface.initialize_active_learning_strategy(strategy=fb_strategy)

interface.start(n_init_labeled=100, n_round=5, n_query=100)
