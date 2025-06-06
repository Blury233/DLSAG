import argparse
import os

def get_main_args():
    """
    Defines and parses command-line arguments for all project scripts.
    """
    parser = argparse.ArgumentParser(
        description="DLSAG: Deep Learning for Steiner Tree Approximation on Graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Main command to execute
    parser.add_argument('command', choices=['generate_data', 'train', 'run'],
                        help="The main task to perform: 'generate_data', 'train', or 'run'.")

    # Group for file paths and directories
    path_args = parser.add_argument_group('Path Arguments')
    path_args.add_argument('--dataset_dir', type=str, default='data/center_score_dataset',
                           help='Directory to save/load the processed PyG dataset.')
    path_args.add_argument('--model_save_path', type=str, default='models/center_scorer_model.pt',
                           help='Path to save the trained GNN model.')

    # Group for data generation
    data_args = parser.add_argument_group('Data Generation Arguments')
    data_args.add_argument('--force_regenerate_data', action='store_true',
                           help='If set, forces regeneration of the dataset even if it exists.')
    data_args.add_argument('--num_graphs_data', type=int, default=500,
                           help='Number of random graphs to generate for the dataset.')
    data_args.add_argument('--n_nodes_min', type=int, default=30, help='Min number of nodes in generated graphs.')
    data_args.add_argument('--n_nodes_max', type=int, default=150, help='Max number of nodes in generated graphs.')
    data_args.add_argument('--edge_prob_min', type=float, default=0.05, help='Min edge probability for graph generation.')
    data_args.add_argument('--edge_prob_max', type=float, default=0.2, help='Max edge probability for graph generation.')
    data_args.add_argument('--n_terms_min', type=int, default=3, help='Min number of terminals in generated graphs.')
    data_args.add_argument('--n_terms_max', type=int, default=15, help='Max number of terminals in generated graphs.')

    # Group for GNN model and training
    train_args = parser.add_argument_group('Model & Training Arguments')
    train_args.add_argument('--gnn_type', type=str, default='gin',
                            choices=['gin', 'gcn', 'gat', 'gatedgcn'],
                            help='The type of GNN layer to use in the model.')
    train_args.add_argument('--num_nearest_terminals', type=int, default=3,
                            help='Number of nearest terminals to consider for node features.')
    train_args.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the GNN.')
    train_args.add_argument('--num_gin_layers', type=int, default=3, help='Number of GIN layers in the GNN.')
    train_args.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    train_args.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    train_args.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')

    # Group for the RZ-GNN algorithm
    algo_args = parser.add_argument_group('Algorithm Arguments')
    algo_args.add_argument('--k_prime', type=int, default=15,
                           help="Top-k' GNN-predicted centers for local search in RZ algorithm.")
    algo_args.add_argument('--m_nearest', type=int, default=5,
                           help='For each candidate center, consider its m-nearest terminals.')
    algo_args.add_argument('--epsilon', type=float, default=1e-9,
                           help='A small float for numerical stability and comparisons.')
    
    # Arguments for the test run in run_algorithm.py
    test_run_args = parser.add_argument_group('Test Run Arguments')
    test_run_args.add_argument('--test_nodes', type=int, default=100, help='Number of nodes for the test graph.')
    test_run_args.add_argument('--test_edge_prob', type=float, default=0.1, help='Edge probability for the test graph.')
    test_run_args.add_argument('--test_terminals', type=int, default=8, help='Number of terminals for the test graph.')
    
    args = parser.parse_args()
    
    # Ensure directories for saving exist
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    return args