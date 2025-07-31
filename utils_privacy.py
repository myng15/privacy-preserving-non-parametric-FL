import os
import argparse
import sys
import numpy as np
import torch
from typing import List
from collections import defaultdict

from utils.privacy_metrics import *


def compute_metrics(base_results_dir: str, seeds: List[int]):
    all_clients_dcr = defaultdict(list)  
    all_clients_dcr_err = defaultdict(list)  
    all_clients_dcr_priv_loss = defaultdict(list)  
    all_clients_dcr_priv_loss_err = defaultdict(list)  
    
    all_clients_ident = defaultdict(list)  
    
    all_clients_mia_precision = defaultdict(list)
    all_clients_mia_recall = defaultdict(list)
    all_clients_mia_f1_macro = defaultdict(list)
    all_clients_mia_auc = defaultdict(list)
    all_clients_mia_acc = defaultdict(list)

    all_clients_domias_precision = defaultdict(list)
    all_clients_domias_recall = defaultdict(list)
    all_clients_domias_f1_macro = defaultdict(list)
    all_clients_domias_auc = defaultdict(list)
    all_clients_domias_acc = defaultdict(list)

    for seed in seeds:
        # Resolve saved_embeddings path for current seed
        seed_dirs = [d for d in os.listdir(base_results_dir) if f"seed={seed}" in d]
        if not seed_dirs:
            print(f"No directory found for seed={seed} in {base_results_dir}")
            continue

        seed_dir = os.path.join(base_results_dir, seed_dirs[0], "saved_embeddings")
        if not os.path.isdir(seed_dir):
            print(f"Directory {seed_dir} not found.")
            continue

        client_files = sorted([
            f for f in os.listdir(seed_dir)
            if f.startswith("client_") and f.endswith(".npz")
        ])
        n_clients = len(client_files)

        print(f"\n[Seed {seed}] Evaluating {n_clients} clients in {seed_dir}...")

        for client_file in client_files:
            client_id = int(client_file.split("_")[1].split(".")[0])
            path = os.path.join(seed_dir, client_file)
            client_embeddings = np.load(path)
            real_embeddings = client_embeddings["real_embeddings"] 
            hout_embeddings = client_embeddings["real_val_embeddings"] 
            generated_embeddings = client_embeddings["generated_embeddings"] 

            # DCR
            dcr_dict = evaluate_dcr(real_embeddings, generated_embeddings, hout_embeddings)
            dcr = dcr_dict["median_dcr"]
            dcr_err = dcr_dict["err_dcr"]
            dcr_priv_loss = dcr_dict["priv_loss"]
            dcr_priv_loss_err = dcr_dict["priv_loss_err"]
            all_clients_dcr[client_id].append(dcr)
            all_clients_dcr_err[client_id].append(dcr_err)
            all_clients_dcr_priv_loss[client_id].append(dcr_priv_loss)
            all_clients_dcr_priv_loss_err[client_id].append(dcr_priv_loss_err)


            # IDENTIFIABILITY SCORE
            ident = evaluate_identifiability_score(real_embeddings, generated_embeddings, distance_metric="entropy") #"euclidean"
            all_clients_ident[client_id].append(ident)

            # MIA
            mia_dict = evaluate_mia(real_embeddings, generated_embeddings, hout_embeddings)
            mia_precision = mia_dict["MIA precision"]
            mia_recall = mia_dict["MIA recall"]
            mia_f1_macro = mia_dict["MIA macro F1"]
            mia_auc = mia_dict["MIA AUC"]
            mia_acc = mia_dict["MIA accuracy"]
            all_clients_mia_precision[client_id].append(mia_precision)
            all_clients_mia_recall[client_id].append(mia_recall)
            all_clients_mia_f1_macro[client_id].append(mia_f1_macro)
            all_clients_mia_auc[client_id].append(mia_auc)
            all_clients_mia_acc[client_id].append(mia_acc)

            # DOMIAS
            domias_dict = evaluate_domias_mia(mem_set=real_embeddings, non_mem_set=hout_embeddings, reference_set=hout_embeddings, anonymized_set=generated_embeddings)
            domias_precision = domias_dict["DOMIAS precision"]
            domias_recall = domias_dict["DOMIAS recall"]
            domias_f1_macro = domias_dict["DOMIAS macro F1"]
            domias_auc = domias_dict["DOMIAS AUC"]
            domias_acc = domias_dict["DOMIAS accuracy"]
            all_clients_domias_precision[client_id].append(domias_precision)
            all_clients_domias_recall[client_id].append(domias_recall)
            all_clients_domias_f1_macro[client_id].append(domias_f1_macro)
            all_clients_domias_auc[client_id].append(domias_auc)
            all_clients_domias_acc[client_id].append(domias_acc)


    # DCR
    print("\n=== Averaged Median DCR metrics per Client ===")
    metrics = {
        "DCR": all_clients_dcr,
        "DCR Standard Error": all_clients_dcr_err,
        "DCR Privacy Loss": all_clients_dcr_priv_loss,
        "DCR Privacy Loss Standard Error": all_clients_dcr_priv_loss_err
    }

    client_avg_metrics = {metric: [] for metric in metrics}

    for metric, client_metric_dict in metrics.items():
        print(f"\n=== {metric.upper()} ===")
        for client_id in sorted(client_metric_dict.keys()):
            scores = client_metric_dict[client_id]
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            client_avg_metrics[metric].append(avg_score)
            print(f"Client_{client_id}: Average {metric} = {avg_score:.2f} (Std: {std_score:.2f})")

        overall_avg = np.mean(client_avg_metrics[metric])
        overall_std = np.std(client_avg_metrics[metric])
        print(f"\nOverall average {metric} across clients: {overall_avg:.2f} (Std: {overall_std:.2f})")


    # IDENTIFIABILITY SCORE
    print("\n=== Averaged Identifiability Score per Client ===")
    client_avg_idents = []
    for client_id in sorted(all_clients_ident.keys()):
        idents = all_clients_ident[client_id]
        avg_ident = np.mean(idents)
        std_ident = np.std(idents)
        client_avg_idents.append(avg_ident)
        print(f"Client_{client_id}: Average Identifiability Score = {avg_ident:.2f} (Std: {std_ident:.2f})")

    overall_avg_ident = np.mean(client_avg_idents)
    overall_std_ident = np.std(client_avg_idents)

    print(f"\nOverall average Identifiability Score across clients: {overall_avg_ident:.2f} (Std: {overall_std_ident:.2f})")


    # MIA
    print("\n=== Averaged MIA metrics per Client ===")
    metrics = {
        "Precision": all_clients_mia_precision,
        "Recall": all_clients_mia_recall,
        "F1 (Macro)": all_clients_mia_f1_macro,
        "ROC-AUC": all_clients_mia_auc,
        "Accuracy": all_clients_mia_acc
    }

    client_avg_metrics = {metric: [] for metric in metrics}

    for metric, client_metric_dict in metrics.items():
        print(f"\n=== {metric.upper()} ===")
        for client_id in sorted(client_metric_dict.keys()):
            scores = client_metric_dict[client_id]
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            client_avg_metrics[metric].append(avg_score)
            print(f"Client_{client_id}: Average MIA {metric} = {avg_score:.2f} (Std: {std_score:.2f})")

        overall_avg = np.mean(client_avg_metrics[metric])
        overall_std = np.std(client_avg_metrics[metric])
        print(f"\nOverall average MIA {metric} across clients: {overall_avg:.2f} (Std: {overall_std:.2f})")

    
    # DOMIAS
    print("\n=== Averaged DOMIAS metrics per Client ===")
    metrics = {
        "Precision": all_clients_domias_precision,
        "Recall": all_clients_domias_recall,
        "F1 (Macro)": all_clients_domias_f1_macro,
        "ROC-AUC": all_clients_domias_auc,
        "Accuracy": all_clients_domias_acc
    }

    client_avg_metrics = {metric: [] for metric in metrics}

    for metric, client_metric_dict in metrics.items():
        print(f"\n=== {metric.upper()} ===")
        for client_id in sorted(client_metric_dict.keys()):
            scores = client_metric_dict[client_id]
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            client_avg_metrics[metric].append(avg_score)
            print(f"Client_{client_id}: Average DOMIAS {metric} = {avg_score:.2f} (Std: {std_score:.2f})")

        overall_avg = np.mean(client_avg_metrics[metric])
        overall_std = np.std(client_avg_metrics[metric])
        print(f"\nOverall average DOMIAS {metric} across clients: {overall_avg:.2f} (Std: {overall_std:.2f})")



if __name__ == "__main__":
    base_results_dirs = [
        # e.g.
        'results/organsmnist/base_patch14_dinov2/s_frac=1.0/10_clients/by_labels_split/alpha=0.3/gen_seed=202502/val_frac=0.1/global_sampling=random/linear_cvae_fedavg/3_layer_CVAE/knn_metric=euclidean/gaussian_kernel/n_neighbors=3/classifier_optimizer=adam/100_local_epochs/anonymizer_lr=0.001/50_fedavg_rounds/dp/mgn=1.5-eps=1.0-delta=1e-4/total_generated_factor=1.0/augmentation=inverse/2025_07_10_19_27',
        'results/organsmnist/base_patch14_dinov2/s_frac=1.0/10_clients/by_labels_split/alpha=0.3/gen_seed=202502/val_frac=0.1/linear_centroids/knn_metric=euclidean/gaussian_kernel/n_neighbors=3/classifier_optimizer=adam/100_local_epochs/k_same=10/unsupervised/return_centroids=True/non_dp/2025_07_13_20_23',
    ]

    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, message):
            for f in self.files:
                f.write(message)
        def flush(self):
            pass

    # Save original stdout so we can restore it
    original_stdout = sys.stdout

    for base_results_dir in base_results_dirs:
        logfile = os.path.join(base_results_dir, "results_all_seeds.txt")
        seeds = [3407, 12345, 202502]

        log_file = open(logfile, "a")
        sys.stdout = TeeLogger(original_stdout, log_file)

        compute_metrics(base_results_dir, seeds)

        # Restore sys.stdout and close log file
        sys.stdout = original_stdout
        log_file.close()


