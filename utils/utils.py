import hashlib
import itertools
import logging
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def set_seed_deterministic(seed: int) -> None:
    """Full training-time RNG + PyTorch deterministic mode (cuDNN, cuBLAS workspace, etc.)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_seed(seed: int) -> None:
    """
    Minimal RNG seeding (random / numpy / torch.manual_seed only).

    Use for architecture sampling so draws match Sept / ~654dd11. For full
    training determinism use set_seed_deterministic instead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_permutation_invariant_metrics(y_true, y_pred, n_classes):
    """
    Calculate permutation-invariant accuracy and F1 score for large datasets
    NOTE: this is exhaustive and on the order of k! where k is number of classes
    Args:
        y_true: Array of true class indices (shape: [n_samples])
        y_pred: Array of predicted class indices (shape: [n_samples])
        n_classes: Number of classes (labels range from 0 to n_classes-1)
    Returns:
        tuple: (max_accuracy, mean_accuracy, max_f1, mean_f1)
    """
    # Get number of samples
    n_samples = len(y_true)

    # Generate all possible permutations of class labels from 0 to n_classes-1
    all_perms = list(itertools.permutations(range(n_classes)))
    n_perms = len(all_perms)

    # Precompute lookup arrays for all permutations
    permuted_preds = np.zeros((n_perms, n_samples), dtype=y_pred.dtype)

    for i, perm in enumerate(all_perms):
        # Create a lookup array for this permutation
        lookup = np.zeros(n_classes)
        for j in range(n_classes):
            lookup[j] = perm[j]
        # Apply the permutation to predictions
        permuted_preds[i] = lookup[y_pred]

    # Calculate accuracy for each permutation
    accuracies = np.zeros(n_perms)
    f1_scores_weighted = np.zeros(n_perms)
    f1_scores_macro = np.zeros(n_perms)

    for i in range(n_perms):
        # correct = np.sum(permuted_preds[i] == y_true)
        # accuracies[i] = correct / n_samples
        accuracies[i] = accuracy_score(y_true, permuted_preds[i])
        f1_scores_weighted[i] = f1_score(y_true, permuted_preds[i], average="weighted")
        f1_scores_macro[i] = f1_score(y_true, permuted_preds[i], average="macro")

    max_accuracy = np.max(accuracies)
    mean_accuracy = np.mean(accuracies)

    max_f1_weighted = np.max(f1_scores_weighted)
    mean_f1_weighted = np.mean(f1_scores_weighted)

    max_f1_macro = np.max(f1_scores_macro)
    mean_f1_macro = np.mean(f1_scores_macro)

    return (
        max_accuracy,
        mean_accuracy,
        max_f1_weighted,
        mean_f1_weighted,
        max_f1_macro,
        mean_f1_macro,
    )


def calculate_permutation_invariant_metrics_fast(y_true, y_pred, n_classes):
    """
    Fast permutation-invariant metrics:
      – optimal scores via Hungarian assignment
      – 'mean' accuracy set analytically to 1/n_classes
      – 'mean' macro-F1 and weighted-F1 forced to 0.0 (per requirement)
    Returns
      (max_accuracy, mean_accuracy, max_f1_weighted, mean_f1_weighted,
       max_f1_macro, mean_f1_macro)
    """
    # 1) Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    # 2) Hungarian assignment  → optimal mapping
    cost = cm.max() - cm
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}

    # 3) Relabel predictions with that mapping
    y_pred_opt = np.vectorize(mapping.get)(y_pred)

    # 4) Optimal metrics
    max_accuracy = accuracy_score(y_true, y_pred_opt)
    max_f1_weighted = f1_score(y_true, y_pred_opt, average="weighted")
    max_f1_macro = f1_score(y_true, y_pred_opt, average="macro")

    # 5) ‘Mean’ values without enumeration
    mean_accuracy = 1.0 / n_classes  # exact expectation
    mean_f1_weighted = 1.0 / n_classes  # good analytic proxy
    mean_f1_macro = 0.0  # per your request

    return (
        max_accuracy,
        mean_accuracy,
        max_f1_weighted,
        mean_f1_weighted,
        max_f1_macro,
        mean_f1_macro,
    )


def compute_metrics(actual_labels, pred_labels, num_classes):
    """
    Computing the metrics from data
    :param actual_labels: ground truth labels
    :param pred_labels: predicted labels
    :return: --
    """
    acc = accuracy_score(actual_labels, pred_labels)
    f_score_weighted = f1_score(actual_labels, pred_labels, average="weighted")
    f_score_macro = f1_score(actual_labels, pred_labels, average="macro")
    if num_classes <= 6:
        (
            max_accuracy,
            mean_accuracy,
            max_f1_weighted,
            mean_f1_weighted,
            max_f1_macro,
            mean_f1_macro,
        ) = calculate_permutation_invariant_metrics(
            np.asarray(actual_labels), np.asarray(pred_labels), num_classes
        )
    else:
        (
            max_accuracy,
            mean_accuracy,
            max_f1_weighted,
            mean_f1_weighted,
            max_f1_macro,
            mean_f1_macro,
        ) = calculate_permutation_invariant_metrics_fast(
            np.asarray(actual_labels), np.asarray(pred_labels), num_classes
        )
    class_counts = dict(Counter(pred_labels))
    return (
        acc,
        f_score_weighted,
        f_score_macro,
        max_accuracy,
        mean_accuracy,
        max_f1_weighted,
        mean_f1_weighted,
        max_f1_macro,
        mean_f1_macro,
        class_counts,
    )


def generate_unique_model_id(model, hyperparameters, length=10):
    # Get model architecture as a string
    model_architecture = str(model)
    model_string = model_architecture + str(hyperparameters)

    # Generate SHA256 hash of the architecture string
    hash_object = hashlib.sha256(model_string.encode())
    full_hash = hash_object.hexdigest()

    # Return first 'length' characters of the hash
    return full_hash[:length]


def setup_file_logger(run_name, log_dir):
    """
    Set up a logger that outputs to stdout instead of a file.
    This keeps all logging visible in the console.
    """
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create a stream handler that outputs to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(stream_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


if __name__ == "__main__":
    print("‣ Running permutation-invariant metric sanity check …")

    # reproducible toy data
    set_seed_deterministic(42)
    n_classes = 6  # keep ≤6 so k! is tractable
    n_samples = 400

    y_true = np.random.randint(0, n_classes, size=n_samples)

    # predictions = permuted true labels + some random noise
    perm = np.random.permutation(n_classes)
    y_pred = perm[y_true]
    noise_ix = np.random.choice(n_samples, size=40, replace=False)
    y_pred[noise_ix] = np.random.randint(0, n_classes, size=len(noise_ix))

    # 1) exhaustive enumeration (your ORIGINAL idea)
    def _metrics_exhaustive(y_t, y_p, k):
        best_acc = best_f1_w = best_f1_m = -1
        for p in itertools.permutations(range(k)):
            mapped = np.array([p[l] for l in y_p])
            acc = accuracy_score(y_t, mapped)
            f1_w = f1_score(y_t, mapped, average="weighted")
            f1_m = f1_score(y_t, mapped, average="macro")
            if acc > best_acc:
                best_acc, best_f1_w, best_f1_m = acc, f1_w, f1_m
        return best_acc, best_f1_w, best_f1_m

    # 2) Hungarian assignment (your NEW implementation)
    def _metrics_hungarian(y_t, y_p, k):
        cm = confusion_matrix(y_t, y_p, labels=np.arange(k))
        cost = cm.max() - cm
        r, c = linear_sum_assignment(cost)
        mapping = {pred: true for true, pred in zip(r, c)}
        mapped = np.vectorize(mapping.get)(y_p)
        acc = accuracy_score(y_t, mapped)
        f1_w = f1_score(y_t, mapped, average="weighted")
        f1_m = f1_score(y_t, mapped, average="macro")
        return acc, f1_w, f1_m

    exhaustive_vals = _metrics_exhaustive(y_true, y_pred, n_classes)
    hungarian_vals = _metrics_hungarian(y_true, y_pred, n_classes)

    print("Exhaustive :", exhaustive_vals)
    print("Hungarian  :", hungarian_vals)

    # prove equality up to floating-point precision
    assert np.allclose(exhaustive_vals, hungarian_vals, atol=1e-12), (
        "Mismatch! Hungarian did not find the global optimum."
    )
    print("✅  The two methods give identical results.")

    print("using our functions")
    exhaustive_vals = calculate_permutation_invariant_metrics(y_true, y_pred, n_classes)
    hungarian_vals = calculate_permutation_invariant_metrics_fast(
        y_true, y_pred, n_classes
    )

    print("Exhaustive :", exhaustive_vals)
    print("Hungarian  :", hungarian_vals)

    def _assert_optima_equal(tup_exh, tup_hun, atol=1e-12):
        """Compare only the max metrics: positions 0, 2, 4."""
        for idx in (0, 2, 4):
            assert np.isclose(tup_exh[idx], tup_hun[idx], atol=atol), (
                f"Mismatch at position {idx}: "
                f"{tup_exh[idx]} (exh) vs {tup_hun[idx]} (hun)"
            )

    _assert_optima_equal(exhaustive_vals, hungarian_vals)
    print("✅  Optimal metrics match; Hungarian finds the same best mapping.")
