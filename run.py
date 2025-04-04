import argparse  # noqa: D100
parser = argparse.ArgumentParser()
parser.add_argument('--fb_strategy', type=str, default="Random")
args = parser.parse_args()

fb_strategy = args.strategy

interface = ALPipe(weight_bias_track = True, 
                     exp_name = strategy,
                     device = 'cuda', 
                     seed = 1)

# data_path
path = './dataset/random_promo'
interface.initialize_data(path = path,
                          dataset_name='sub_sample_pTpA_All',
                          batch_size = 32)

interface.initialize_model(epochs = 20, hidden_size = 64)
interface.initialize_active_learning_strategy(strategy = strategy)

interface.start(n_init_labeled = 100, n_round = 5, n_query = 100)