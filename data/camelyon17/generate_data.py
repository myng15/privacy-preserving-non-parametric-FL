from collections import Counter
import os
import argparse
import warnings

import numpy as np

import sys
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(current))
# adding the root directory to the sys.path.
sys.path.append(root)

from sklearn.model_selection import train_test_split

from utils.utils import seed_everything
  
PATH = os.path.join(root, "data/camelyon17/all_clients_data/") 
RAW_DATA_PATH = os.path.join(root, "data/camelyon17/database/")


def subsample(embeddings, labels, s_frac, seed, n_clients=None):
    """
    sample embeddings if not using the entire dataset

    :param indices:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if s_frac == 1.0:
        warnings.warn("Subsampling is not needed (s_frac == 1.0)", RuntimeWarning)
        return
    
    np.random.seed(seed)
    indices = range(len(embeddings))
    
    if s_frac < 1.0:
        n_samples = int(len(embeddings) * s_frac)
        selected_indices = np.random.choice(indices, size=n_samples, replace=False)
    
    elif s_frac > 1.0: # CUSTOM SAMPLE SIZE IN TOTAL
        n_samples = int(s_frac / n_clients) # CUSTOM SAMPLE SIZE PER CLIENT
        selected_indices = np.random.choice(indices, size=n_samples, replace=False)

    return embeddings[selected_indices], labels[selected_indices]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Camelyon17 clients split based on hospitals/medical centers that the data come from.'
    )
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--n_clients',
        help='number of clients;',
        type=int,
        required=True
    )
    parser.add_argument(
        '--test_clients_frac',
        help='fraction of clients not participating in the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--test_data_frac',
        help='fraction of data used for test in each client\'s allocated data; default is 0.2',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--val_data_frac',
        help='fraction of data used for validation in each client\'s allocated data; default is 0.2',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=12345
    )

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    all_clients_ids = range(args.n_clients)
    if args.test_clients_frac > 0:
        train_clients_ids, test_clients_ids = train_test_split(
            all_clients_ids, test_size=args.test_clients_frac, random_state=args.seed
        )
    else: 
        train_clients_ids, test_clients_ids = all_clients_ids, []

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    # Process clients
    for mode, client_ids in [("train", train_clients_ids), ("test", test_clients_ids)]:

        for client_id in client_ids:
            file_path = os.path.join(RAW_DATA_PATH, f"client_{client_id}.npz")
            if not os.path.exists(file_path):
                print(f"File {file_path} not found. Skipping.")
                continue

            # Load the dataset
            data = np.load(file_path)
            embeddings = data['embeddings']
            labels = data['labels']

            if args.s_frac < 1.0:
                embeddings, labels = subsample(embeddings, labels, args.s_frac, args.seed)
            elif args.s_frac > 1.0 : #CUSTOM SAMPLE SIZE IN TOTAL
                embeddings, labels = subsample(embeddings, labels, args.s_frac, args.seed, args.n_clients)

            # Perform stratified train-test split
            train_emb, test_emb, train_labels, test_labels = train_test_split(
                embeddings, labels, 
                test_size=args.test_data_frac, 
                stratify=labels, 
                random_state=args.seed
            )

            # Perform stratified train-val split on the training set
            train_emb_final, val_emb, train_labels_final, val_labels = train_test_split(
                train_emb, train_labels, 
                test_size=args.val_data_frac / (1 - args.test_data_frac), # val_frac / (1 - test_frac) is the portion for val set from the remaining portion after splitting off the test set from the entire dataset
                stratify=train_labels, 
                random_state=args.seed 
            )

            # Create the output directory
            client_dir = os.path.join(PATH, mode, f"client_{client_id}")
            os.makedirs(client_dir, exist_ok=True)

            # Save the splits
            np.savez(os.path.join(client_dir, "train.npz"), embeddings=train_emb_final, labels=train_labels_final)
            np.savez(os.path.join(client_dir, "val.npz"), embeddings=val_emb, labels=val_labels)
            np.savez(os.path.join(client_dir, "test.npz"), embeddings=test_emb, labels=test_labels)

            print(f"Client {client_id} ({mode} mode): train/val/test splits saved.")

if __name__ == "__main__":
    main()


