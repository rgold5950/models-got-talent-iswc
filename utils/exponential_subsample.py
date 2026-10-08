import numpy as np

def _exponential_subsample(
    data: np.ndarray,
    labels: np.ndarray,
    rho: float,
    seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample each class so that:
    - largest class keeps MAX_SIZE
    - smallest class ends at MAX_SIZE * rho
    - intermediate classes follow MAX_SIZE * exp(beta * idx)
    
    Here, rho = smallest_count / largest_count (0 < rho <= 1), and
    beta = log(rho) / (C - 1), where C is the number of classes.
    """
    # 1. raw counts & sort descending
    classes, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    classes = classes[order]
    counts = counts[order]
    
    C = len(classes)
    MAX_SIZE = counts[0]
    min_size = MAX_SIZE * rho
    if counts[-1] < min_size:
        raise ValueError(
            f"ρ={rho:.3f} too large: smallest class only has {counts[-1]} samples, "
            f"needs ≥ {int(np.ceil(min_size))}."
        )

    beta = np.log(rho) / (C - 1)
    targets = {
        cls: int(np.round(MAX_SIZE * np.exp(beta * idx)))
        for idx, cls in enumerate(classes)
    }
    assert targets[classes[0]] == MAX_SIZE
    assert targets[classes[-1]] == int(np.round(min_size))
    rng = np.random.default_rng(seed)
    picks = []
    for cls in classes:
        idxs = np.where(labels == cls)[0]
        n_target = targets[cls]
        picks.append(rng.choice(idxs, size=n_target, replace=False))
    all_idx = rng.permutation(np.concatenate(picks))
    return data[all_idx], labels[all_idx]