import glob
import os
import pickle
import numpy as np
from collections import Counter
import argparse


def load_client_results(results_dir, seed):
    path = os.path.join(results_dir, f"seed={seed}-chkpt=*/train/client_results_dict.pkl")
    
    # Resolve wildcard path (if any)
    possible_files = sorted(glob.glob(path))
    if not possible_files:
        print(f"Warning: No results found for seed {seed} in {results_dir}")
        return None
    
    with open(possible_files[-1], "rb") as f:
        return pickle.load(f)

def process_results(results_dir, seeds):
    all_seeds_results = []
    
    for seed in seeds:
        client_results_dict = load_client_results(results_dir, seed)
        if client_results_dict is None:
            continue

        seed_results = []
        for client_id, client_info in client_results_dict.items():
            seed_results.append({
                "client_id": client_id,
                "train_samples": client_info['n_train_samples'],
                "accuracies": client_info['test_acc'],
                "balanced_accuracies": client_info['test_balanced_acc'],
                "auc_scores": client_info['test_auc'],
            })

        all_seeds_results.append(seed_results)

    if not all_seeds_results:
        print("No results available to process.")
        return

    client_results = {}
    for seed_result in all_seeds_results:
        for client in seed_result:
            client_id = client["client_id"]
            if client_id not in client_results:
                client_results[client_id] = {
                    "train_samples": client["train_samples"],
                    "accuracies": [],
                    "balanced_accuracies": [],
                    "auc_scores": []
                }
            client_results[client_id]["accuracies"].append(client["accuracies"])
            client_results[client_id]["balanced_accuracies"].append(client["balanced_accuracies"])
            client_results[client_id]["auc_scores"].append(client["auc_scores"])


    final_results = []
    for client_id, data in sorted(client_results.items(), key=lambda x: x[1]["train_samples"]):
        avg_acc = np.mean(data["accuracies"])
        avg_bal_acc = np.mean(data["balanced_accuracies"])
        avg_auc = np.mean(data["auc_scores"])
        final_results.append((data["train_samples"], avg_acc, avg_bal_acc, avg_auc))


    print("\nClient results (sorted by train samples):")
    for samples, avg_acc, avg_bal_acc, avg_auc in final_results:
        print(f"Train samples: {samples}, Average Accuracy: {avg_acc:.2f}%, Average Balanced Accuracy: {avg_bal_acc:.2f}%, Average ROC-AUC: {avg_auc:.2f}")

    overall_avg_acc = np.mean([res[1] for res in final_results])
    overall_std_acc = np.std([res[1] for res in final_results])
    overall_avg_bal_acc = np.mean([res[2] for res in final_results])
    overall_std_bal_acc = np.std([res[2] for res in final_results])
    overall_avg_auc = np.mean([res[3] for res in final_results])
    overall_std_auc = np.std([res[3] for res in final_results])

    print(f"\nOverall average accuracy across clients: {overall_avg_acc:.2f}% (Std: {overall_std_acc:.2f}%)")
    print(f"Overall average balanced accuracy across clients: {overall_avg_bal_acc:.2f}% (Std: {overall_std_bal_acc:.2f}%)")
    print(f"Overall average ROC-AUC across clients: {overall_avg_auc:.2f} (Std: {overall_std_auc:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results from federated learning experiments.")
    parser.add_argument("--results_dir", type=str, required=True, help="Base directory for results")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="List of seed values")

    args = parser.parse_args()
    process_results(args.results_dir, args.seeds)
