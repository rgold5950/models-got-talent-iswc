"""
Strict validation for experiment configs used with Ray Tune.

No implicit defaults or scaling: num_models and experiments.model_type_counts
must agree exactly with the YAML (after unified-config merge).
"""

from __future__ import annotations

from typing import Any, Dict

REQUIRED_TOP_LEVEL = (
    "dataset",
    "model_types",
    "num_models",
    "hyperparameters",
    "model_type_counts",
)

REQUIRED_SCALAR_HP_KEYS = (
    "batch_size",
    "learning_rate",
    "weight_decay",
    "num_epochs",
    "step_size",
    "gamma",
)


def validate_experiments_for_tune(exp: Dict[str, Any]) -> None:
    """
    Validate merged `cfg["experiments"]` before building a Tune search space.

    Raises:
        ValueError: if the config is incomplete, inconsistent, or uses unsupported shapes (e.g. list grids).
    """
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in exp]
    if missing:
        raise ValueError(f"experiments config missing required keys: {missing}")

    model_types = exp["model_types"]
    if not isinstance(model_types, list) or len(model_types) == 0:
        raise ValueError("experiments.model_types must be a non-empty list")

    num_models = exp["num_models"]
    if not isinstance(num_models, int) or num_models < 1:
        raise ValueError("experiments.num_models must be a positive integer")

    hp = exp["hyperparameters"]
    if not isinstance(hp, dict):
        raise ValueError("experiments.hyperparameters must be a dictionary")

    for k in REQUIRED_SCALAR_HP_KEYS:
        if k not in hp:
            raise ValueError(f"hyperparameters missing required key: {k!r}")
        v = hp[k]
        if isinstance(v, (list, tuple)):
            raise ValueError(
                f"hyperparameters[{k!r}] must be a scalar for Tune runs; "
                f"got a sequence (grid search is not supported in strict mode)."
            )

    mtc = exp["model_type_counts"]
    if not isinstance(mtc, dict) or not mtc:
        raise ValueError("experiments.model_type_counts must be a non-empty dictionary")

    types_list = list(model_types)
    types_set = set(types_list)
    if len(types_list) != len(types_set):
        raise ValueError("experiments.model_types must not contain duplicates")

    counts_set = set(mtc.keys())
    if types_set != counts_set:
        raise ValueError(
            f"model_types {sorted(types_set)} must exactly match model_type_counts keys "
            f"{sorted(counts_set)} (same set, no extras or omissions)."
        )

    counts = []
    for mt in types_list:
        c = mtc[mt]
        if not isinstance(c, int) or c < 1:
            raise ValueError(
                f"model_type_counts[{mt!r}] must be a positive integer, got {c!r}"
            )
        counts.append(c)

    total = sum(counts)
    if total != num_models:
        raise ValueError(
            f"num_models ({num_models}) must equal sum(model_type_counts) ({total}). "
            f"Per-type counts: {dict(zip(types_list, counts))}"
        )

