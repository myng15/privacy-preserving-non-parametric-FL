import os
import pickle
from typing import List
import numpy as np
from collections import Counter

import glob
import argparse


def load_results(results_dir, seed):
    path = os.path.join(results_dir, f"seed={seed}-chkpt=*/results_dict.pkl")

    # Resolve wildcard path (if any)
    possible_files = sorted(glob.glob(path))
    if not possible_files:
        print(f"Warning: No results found for seed {seed} in {results_dir}")
        return None

    with open(possible_files[-1], "rb") as f:
        return pickle.load(f)


def process_results(results_dir: str, seeds: List[int]):
    all_seeds_results = {}

    for seed in seeds:
        results_dict = load_results(results_dir, seed)
        if results_dict is None:
            continue

        all_client_ids = results_dict['all_client_ids']
        all_test_scores = results_dict['all_test_scores']
        all_test_balanced_acc_scores = results_dict['all_test_balanced_acc_scores']
        all_test_auc_scores = results_dict['all_test_auc_scores']
        all_test_f1_scores_macro = results_dict['all_test_f1_scores_macro']
        all_test_f1_scores_weighted = results_dict['all_test_f1_scores_weighted'] 
        n_train_samples = results_dict['n_train_samples']
        n_test_samples = results_dict['n_test_samples']
        all_wasserstein_dist = results_dict['all_wasserstein_dist']

        if len(all_client_ids) == 0:
            all_client_ids = np.array([0, 1, 2, 3, 4])
        
        individual_accuracies = (all_test_scores / n_test_samples) * 100
        individual_balanced_accuracies = (all_test_balanced_acc_scores / n_test_samples) * 100
        individual_auc_scores = (all_test_auc_scores / n_test_samples)
        individual_f1_scores_macro = (all_test_f1_scores_macro / n_test_samples) 
        individual_f1_scores_weighted = (all_test_f1_scores_weighted / n_test_samples) 

        for i, client_id in enumerate(all_client_ids):
            if client_id not in all_seeds_results:
                all_seeds_results[client_id] = {
                    "train_samples": n_train_samples[i],
                    "accuracies": [],
                    "balanced_accuracies": [],
                    "auc_scores": [],
                    "f1_scores_macro": [], # F1
                    "f1_scores_weighted": [], # F1
                    "all_wasserstein_dist": []
                }
            all_seeds_results[client_id]["accuracies"].append(individual_accuracies[i])
            all_seeds_results[client_id]["balanced_accuracies"].append(individual_balanced_accuracies[i])
            all_seeds_results[client_id]["auc_scores"].append(individual_auc_scores[i])
            all_seeds_results[client_id]["f1_scores_macro"].append(individual_f1_scores_macro[i]) 
            all_seeds_results[client_id]["f1_scores_weighted"].append(individual_f1_scores_weighted[i]) 
            if len(all_wasserstein_dist) > 0:
                all_seeds_results[client_id]["all_wasserstein_dist"].append(all_wasserstein_dist[i])

    if not all_seeds_results:
        print("No results available to process.")
        return

    final_results = []
    for client_id, data in sorted(all_seeds_results.items(), key=lambda x: x[1]["train_samples"]):
        avg_acc = np.mean(data["accuracies"])
        avg_bal_acc = np.mean(data["balanced_accuracies"])
        avg_auc = np.mean(data["auc_scores"])
        avg_f1_macro = np.mean(data["f1_scores_macro"]) 
        avg_f1_weighted = np.mean(data["f1_scores_weighted"]) 
        avg_wass_dist = np.mean(data["all_wasserstein_dist"]) if len(data["all_wasserstein_dist"]) > 0 else 0
        final_results.append((data["train_samples"], avg_acc, avg_bal_acc, avg_auc, avg_f1_macro, avg_f1_weighted, avg_wass_dist)) 

    print("\nClient results (sorted by train samples):")
    for samples, avg_acc, avg_bal_acc, avg_auc, avg_f1_macro, avg_f1_weighted, wass_dist in final_results: 
        print(f"Train samples: {samples}, Average Accuracy: {avg_acc:.2f}%, Average Balanced Accuracy: {avg_bal_acc:.2f}%, Average ROC-AUC: {avg_auc:.6f}, Average F1 (Macro): {avg_f1_macro:.6f}, Average F1 (Weighted): {avg_f1_weighted:.6f}, Average Wasserstein dist.: {wass_dist:.6f}") 

    overall_avg_acc = np.mean([res[1] for res in final_results])
    overall_std_acc = np.std([res[1] for res in final_results])
    overall_avg_bal_acc = np.mean([res[2] for res in final_results])
    overall_std_bal_acc = np.std([res[2] for res in final_results])
    overall_avg_auc = np.mean([res[3] for res in final_results])
    overall_std_auc = np.std([res[3] for res in final_results])

    overall_avg_f1_macro = np.mean([res[4] for res in final_results])
    overall_std_f1_macro = np.std([res[4] for res in final_results])
    overall_avg_f1_weighted = np.mean([res[5] for res in final_results])
    overall_std_f1_weighted = np.std([res[5] for res in final_results])

    overall_avg_wass_dist = np.mean([res[6] for res in final_results])
    overall_std_wass_dist = np.std([res[6] for res in final_results])

    print(f"\nOverall average accuracy across clients: {overall_avg_acc:.2f}% (Std: {overall_std_acc:.2f}%)")
    print(f"Overall average balanced accuracy across clients: {overall_avg_bal_acc:.2f}% (Std: {overall_std_bal_acc:.2f}%)")
    print(f"Overall average ROC-AUC across clients: {overall_avg_auc:.6f} (Std: {overall_std_auc:.6f})")
    print(f"Overall average F1 score (Macro) across clients: {overall_avg_f1_macro:.6f} (Std: {overall_std_f1_macro:.6f})") # F1
    print(f"Overall average F1 score (Weighted) across clients: {overall_avg_f1_weighted:.6f} (Std: {overall_std_f1_weighted:.6f})") # F1
    print(f"Overall average Wasserstein dist. across clients: {overall_avg_wass_dist:.6f} (Std: {overall_std_wass_dist:.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process federated learning experiment results.")
    parser.add_argument("--results_dir", type=str, required=True, help="Base directory for results")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="List of seed values")

    args = parser.parse_args()
    process_results(args.results_dir, args.seeds)

