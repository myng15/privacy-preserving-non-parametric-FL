import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import KBinsDiscretizer


def evaluate_dcr(real_data, synth_data, hout_data=None, metric='euclidean'):
    """
    Compute Distance to Closest Record (DCR) and optional holdout-based privacy loss.

    Parameters:
        real_data (np.ndarray): Real training embeddings.
        synth_data (np.ndarray): Synthetic/anonymized embeddings.
        hout_data (np.ndarray or None): Holdout embeddings (not in training set).
        metric (str): Distance metric for nearest neighbors (default: 'euclidean').

    Returns:
        dict with median DCR and DCR privacy loss results
    """
    def nn_distance(X_ref, Y_query):
        # Fit NearestNeighbors with adjusted neighbor count to skip self-matches
        n_neighbors = 2 if np.array_equal(X_ref, Y_query) else 1
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(X_ref)
        distances, _ = nn.kneighbors(Y_query)
        if np.array_equal(X_ref, Y_query):
            dists = distances[:, 1]  # skip self-match
        else:
            dists = distances[:, 0]
        return dists

    # Core distances
    dist_synth_to_real = nn_distance(real_data, synth_data)
    dist_real_to_real  = nn_distance(real_data, real_data)

    mut_nn = np.median(dist_synth_to_real)
    int_nn = np.median(dist_real_to_real)

    if (int_nn == 0 and mut_nn == 0): dcr = 1
    elif (int_nn == 0 and mut_nn != 0): dcr = 0
    else: dcr = mut_nn / int_nn

    # Standard error
    err_dcr = dcr * np.sqrt(
        (np.std(dist_synth_to_real, ddof=1) / (np.sqrt(len(dist_synth_to_real)) * mut_nn))**2 +
        (np.std(dist_real_to_real, ddof=1) / (np.sqrt(len(dist_real_to_real)) * int_nn))**2
    )

    results = {
        'median_dcr': dcr,
        'err_dcr': err_dcr
    }

    # Optional: compute priv_loss against holdout
    if hout_data is not None:
        dist_synth_to_hout = nn_distance(hout_data, synth_data)
        hout_median = np.median(dist_synth_to_hout)
        priv_loss = mut_nn - hout_median

        err_hout = np.std(dist_synth_to_hout, ddof=1) / np.sqrt(len(dist_synth_to_hout))
        err_mut  = np.std(dist_synth_to_real, ddof=1) / np.sqrt(len(dist_synth_to_real))

        priv_loss_err = np.sqrt(err_mut**2 + err_hout**2)

        results['priv_loss'] = priv_loss
        results['priv_loss_err'] = priv_loss_err

    return results


def compute_inverse_variance_weights(data: np.ndarray) -> np.ndarray:
    """Compute inverse variance weights for each dimension."""
    variances = np.var(data, axis=0)
    weights = 1.0 / (variances + 1e-8)  
    return weights

def compute_entropy_weights(data: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Compute entropy-based weights per feature dimension."""
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    binned = discretizer.fit_transform(data)
    entropies = []
    n_samples = data.shape[0]
    
    for i in range(data.shape[1]):
        _, counts = np.unique(binned[:, i], return_counts=True)
        probs = counts / n_samples
        entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Shannon entropy
        entropies.append(entropy)
    
    entropies = np.array(entropies)
    weights = 1.0 / (entropies + 1e-8)  # Inverse entropy
    return weights

def evaluate_identifiability_score(
    real: np.ndarray,
    synth: np.ndarray,
    distance_metric: str = "euclidean",  
    n_bins_entropy: int = 10
) -> float:
    """
    Compute (un)identifiability score between real and synthetic/anonymized embeddings.
    
    Parameters:
    - real: Real embeddings (n_samples, n_features)
    - synth: Synthetic/anonymized embeddings (n_samples, n_features)
    - distance_metric: Which distance metric to use
    - n_bins_entropy: Number of bins to use for entropy-based weighting
    
    Returns:
        identifiability_score: Fraction of real points closer to synthetic than to 2nd nearest real point, or
        unidentifiability_score: 1 - identifiability_score (indicating privacy)
    """

    assert real.shape[1] == synth.shape[1], "Dimensionality mismatch"

    if distance_metric == "euclidean":
        nbr_real = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(real)
        dist_real, _ = nbr_real.kneighbors(real)
        r_i = dist_real[:, 1]

        nbr_synth = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(synth)
        d_hat = nbr_synth.kneighbors(real, n_neighbors=1)[0].reshape(-1)

    elif distance_metric == "entropy":
        weights = compute_entropy_weights(real, n_bins=n_bins_entropy)
        W = np.sqrt(weights)

        real_w = real * W
        synth_w = synth * W

        nbr_real = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(real_w)
        r_i = nbr_real.kneighbors(real_w)[0][:, 1]

        nbr_synth = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(synth_w)
        d_hat = nbr_synth.kneighbors(real_w)[0].reshape(-1)

    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")

    identifiability_score = float((d_hat < r_i).mean())
    #return identifiability_score # IF RETURN IDENTIFIABILITY SCORE
    unidentifiability_score = 1.0 - identifiability_score
    return unidentifiability_score


def evaluate_mia(real_data, synt_data, hout_data, num_eval_iter=5):
    """
    Evaluate Membership Inference Attack based only on anonymized/synthetic embeddings (no target model).

    Parameters:
        real_data: np.ndarray [N_real, embedding_dim] - real training embeddings 
        synt_data: np.ndarray [N_synt, embedding_dim] - anonymized/synthetic embeddings
        hout_data: np.ndarray [N_holdout, embedding_dim] - real embeddings NOT in training set
        num_eval_iter: int - number of repetitions for robustness

    Returns:
        a dictionary with MIA precision, recall, macro F1, and standard errors
    """

    assert all(isinstance(d, np.ndarray) for d in [real_data, synt_data, hout_data])
    
    metrics = {
        "precision": [], "recall": [], "f1": [], "auc": [], "accuracy": []
    }

    for _ in range(num_eval_iter):
        # Split holdout into train/test for MIA model
        hout_train, hout_test = train_test_split(hout_data, test_size=0.25, random_state=None)
        
        # Create synthetic(1)-vs-holdout(0) training set
        n_train = min(len(hout_train), len(synt_data))
        syn_train = synt_data[np.random.choice(len(synt_data), n_train, replace=False)]
        h_train = hout_train[np.random.choice(len(hout_train), n_train, replace=False)]

        X_train = np.vstack([syn_train, h_train])
        y_train = np.array([1] * n_train + [0] * n_train)

        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        # Create test set: CVAE-training data (1) vs holdout (0)
        n_test = min(len(real_data), len(hout_test))
        real_test = real_data[np.random.choice(len(real_data), n_test, replace=False)]
        h_test = hout_test[np.random.choice(len(hout_test), n_test, replace=False)]

        X_test = np.vstack([real_test, h_test])
        y_test = np.array([1] * n_test + [0] * n_test)

        # Train MIA classifier
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced')

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, average="macro"))
        metrics["auc"].append(roc_auc_score(y_test, y_proba))
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))

    return {
        "MIA precision": np.mean(metrics["precision"]),
        "MIA precision std": np.std(metrics["precision"], ddof=1) / np.sqrt(num_eval_iter),
        "MIA recall": np.mean(metrics["recall"]),
        "MIA recall std": np.std(metrics["recall"], ddof=1) / np.sqrt(num_eval_iter),
        "MIA macro F1": np.mean(metrics["f1"]),
        "MIA macro F1 std": np.std(metrics["f1"], ddof=1) / np.sqrt(num_eval_iter),
        "MIA AUC": np.mean(metrics["auc"]),
        "MIA AUC std": np.std(metrics["auc"], ddof=1) / np.sqrt(num_eval_iter),
        "MIA accuracy": np.mean(metrics["accuracy"]),
        "MIA accuracy std": np.std(metrics["accuracy"], ddof=1) / np.sqrt(num_eval_iter),
    }


def evaluate_domias_mia(
    mem_set: np.ndarray,
    non_mem_set: np.ndarray,
    reference_set: np.ndarray,
    anonymized_set: np.ndarray,
    bandwidth: float = 0.2,
) -> dict:
    """
    Evaluate membership inference attacks as a privacy metric for an anonymized dataset using DOMIAS.

    Args:
        mem_set: np.ndarray [N_mem, embedding_dim] - Real training DINOv2-extracted features.
        non_mem_set: np.ndarray [N_non_mem, embedding_dim] - Real DINOv2-extracted features NOT used in training.
        reference_set: np.ndarray [N_ref, embedding_dim] - Hold-out DINOv2-extracted features.
        anonymized_set: np.ndarray [N_anonym, embedding_dim] - Anonymized/Synthetic DINOv2-extracted features.
        bandwidth (float): Bandwidth parameter for the kernel density estimation.

    Returns:
        a dictionary with the AUCROC and accuracy scores for the attack.
    """
    # Combine member and non-member sets for evaluation
    X_test = np.concatenate([mem_set, non_mem_set])
    Y_test = np.concatenate([np.ones(len(mem_set)), np.zeros(len(non_mem_set))])

    # Determine KDE bandwidth if not given (Silverman's rule of thumb)
    if bandwidth is None:
        std_dev = np.std(anonymized_set, axis=0).mean()
        n = len(anonymized_set)
        bandwidth = 1.06 * std_dev * n ** (-1 / 5)

    # KDE for anonymized (synthetic) data: estimates P_G
    kde_anonymized = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_anonymized.fit(anonymized_set)

    # KDE for reference data: estimates P_R
    kde_reference = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_reference.fit(reference_set)

    # Estimate density scores
    p_G_evaluated = np.exp(kde_anonymized.score_samples(X_test))
    p_R_evaluated = np.exp(kde_reference.score_samples(X_test))

    # Compute the relative density ratio
    p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

    # Predicted membership (threshold = median of p_rel)
    preds = p_rel > np.median(p_rel)

    # Evaluate metrics
    auc = roc_auc_score(Y_test, p_rel)
    acc = accuracy_score(Y_test, preds)
    precision = precision_score(Y_test, preds, zero_division=0)
    recall = recall_score(Y_test, preds, zero_division=0)
    f1 = f1_score(Y_test, preds, average='macro', zero_division=0)

    return {
        "DOMIAS AUC": auc,
        "DOMIAS accuracy": acc,
        "DOMIAS precision": precision,
        "DOMIAS recall": recall,
        "DOMIAS macro F1": f1,
    }



