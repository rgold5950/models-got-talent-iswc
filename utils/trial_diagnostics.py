"""
Diagnostic utilities for calculating and displaying trial counts before running experiments.
"""
import logging
from typing import Dict, List, Tuple, Any, Optional

import torch

from models.model_generator_registry import model_generator_registry
from utils.utils import set_seed

DEFAULT_CNN_PREFLIGHT = {
    "max_channels": 768,
    "max_sum_channels": 3000,
    "max_layers": 5,
    "hard_channel_cutoff": 900,
    "max_channel_kernel_product_sum": 12000,
}


def calculate_trial_counts(param_space: Dict[str, Any], exp_cfg) -> Tuple[int, Dict[str, int]]:
    """
    Return trial totals from ExperimentConfig after build_search_space().

    Strict mode: no inference from Ray Tune objects or uniform-split fallbacks.
    """
    del param_space  # API compatibility only
    if not exp_cfg.model_type_counts or not isinstance(exp_cfg.model_type_counts, dict):
        raise RuntimeError(
            "Trial counts require ExperimentConfig.build_search_space() first "
            "(model_type_counts is unset)."
        )
    total_trials = sum(exp_cfg.model_type_counts.values())
    return total_trials, exp_cfg.model_type_counts.copy()


def display_trial_summary(
    dataset_name: str,
    total_trials: int,
    model_type_breakdown: Dict[str, int],
    test_mode: bool = False
) -> None:
    """
    Display a formatted summary of trial counts.
    
    Args:
        dataset_name: Name of the dataset
        total_trials: Total number of trials
        model_type_breakdown: Dict mapping model_type -> count
        test_mode: Whether running in test mode
    """
    print("\n" + "="*70)
    print("📊 TRIAL COUNT SUMMARY")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Mode: {'🧪 TEST' if test_mode else '🚀 PRODUCTION'}")
    print(f"\nTotal Trials: {total_trials}")
    print("\nBreakdown by Model Type:")
    print("-" * 70)
    
    for model_type, count in sorted(model_type_breakdown.items()):
        percentage = (count / total_trials * 100) if total_trials > 0 else 0
        print(f"  {model_type:20s}: {count:4d} trials ({percentage:5.1f}%)")
    
    print("="*70)


def _infer_sequence_length(sample_input_shape: Optional[Tuple[int, ...]]) -> Optional[int]:
    """Infer temporal sequence length from a single dataset sample shape."""
    if not sample_input_shape:
        return None
    if len(sample_input_shape) == 1:
        return int(sample_input_shape[0])
    if len(sample_input_shape) >= 2:
        return int(sample_input_shape[0])
    return None


def estimate_param_count(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return int(sum(p.numel() for p in model.parameters()))


def estimate_memory_gb(
    model_type: str,
    arch_config: Any,
    param_count: int,
    batch_size: int,
    sequence_length: Optional[int],
) -> float:
    """
    Coarse memory estimate in GB for one training step.
    Includes params + optimizer + grads + rough activations.
    """
    # params + grads + Adam states ~= 16 bytes / parameter
    param_plus_opt_bytes = float(param_count) * 16.0
    activation_bytes = 0.0
    seq_len = max(1, int(sequence_length)) if sequence_length else 128

    if model_type == "CNN" and isinstance(arch_config, list):
        curr_len = seq_len
        for layer_cfg in arch_config:
            out_ch = int(layer_cfg.get("out_channels", 1))
            kernel = int(layer_cfg.get("kernel_size", 1))
            stride = max(1, int(layer_cfg.get("stride", 1)))
            curr_len = max(1, (curr_len - kernel) // stride + 1)
            activation_bytes += batch_size * out_ch * curr_len * 4.0
        # Backward + transient buffers
        activation_bytes *= 3.0

    elif model_type == "RNN" and isinstance(arch_config, list):
        hidden_sum = sum(int(layer_cfg.get("hidden_size", 1)) for layer_cfg in arch_config)
        # LSTM states are heavier than plain activations
        activation_bytes = batch_size * seq_len * hidden_sum * 4.0 * 6.0

    elif model_type == "TRANSFORMER" and isinstance(arch_config, dict):
        d_model = int(arch_config.get("d_model", 64))
        n_layers = int(arch_config.get("num_layers", 2))
        nhead = max(1, int(arch_config.get("nhead", 2)))
        # Token activations
        token_acts = batch_size * seq_len * d_model * n_layers * 4.0 * 4.0
        # Attention score tensors (B, heads, T, T) per layer
        attn_scores = batch_size * nhead * (seq_len ** 2) * n_layers * 4.0
        activation_bytes = token_acts + attn_scores

    else:
        # Generic fallback
        activation_bytes = batch_size * seq_len * 256 * 4.0

    total_bytes = param_plus_opt_bytes + activation_bytes
    return total_bytes / (1024 ** 3)


def _is_cnn_arch_config(arch_config: Any) -> bool:
    return isinstance(arch_config, list) and all(isinstance(layer, dict) for layer in arch_config)


def check_cnn_preflight(arch_config: Any, thresholds: Optional[Dict[str, int]] = None) -> List[str]:
    """
    Validate CNN architecture against explicit anti-blowup constraints.
    Returns violation reasons; empty list means pass.
    """
    if not _is_cnn_arch_config(arch_config):
        return []

    t = DEFAULT_CNN_PREFLIGHT.copy()
    if thresholds:
        t.update(thresholds)

    channels = [int(layer.get("out_channels", 0)) for layer in arch_config]
    kernels = [int(layer.get("kernel_size", 0)) for layer in arch_config]

    violations: List[str] = []
    if channels and max(channels) > int(t["max_channels"]):
        violations.append(f"max(channels)={max(channels)} > {int(t['max_channels'])}")
    if sum(channels) > int(t["max_sum_channels"]):
        violations.append(f"sum(channels)={sum(channels)} > {int(t['max_sum_channels'])}")
    if len(channels) > int(t["max_layers"]):
        violations.append(f"num_layers={len(channels)} > {int(t['max_layers'])}")
    if any(c >= int(t["hard_channel_cutoff"]) for c in channels):
        violations.append(
            f"channel>=hard_cutoff({int(t['hard_channel_cutoff'])}) present"
        )
    channel_kernel_product_sum = sum(c * k for c, k in zip(channels, kernels))
    if channel_kernel_product_sum > int(t["max_channel_kernel_product_sum"]):
        violations.append(
            f"sum(c*k)={channel_kernel_product_sum} > {int(t['max_channel_kernel_product_sum'])}"
        )
    return violations


def merge_arch_guardrails_for_type(
    raw: Optional[Dict[str, Any]], model_type: str
) -> Dict[str, Any]:
    """
    Flatten experiments.arch_guardrails (default + per_model_type) for one model_type.
    Used for diagnostic checks only; same merge as the former train-time resolver.
    """
    if not raw or not isinstance(raw, dict):
        return {}

    default_limits = raw.get("default", {}) if isinstance(raw.get("default"), dict) else {}
    per_type = raw.get("per_model_type", {}) if isinstance(raw.get("per_model_type"), dict) else {}
    type_limits = per_type.get(model_type, {}) if isinstance(per_type.get(model_type), dict) else {}

    if default_limits or per_type:
        limits: Dict[str, Any] = dict(default_limits)
        limits.update(type_limits)
        return {
            k: v
            for k, v in limits.items()
            if k in ("max_params", "max_estimated_memory_gb", "cnn_preflight")
        }

    # Legacy flat dict (no nested default/per_model_type)
    return {
        k: raw[k]
        for k in ("max_params", "max_estimated_memory_gb", "cnn_preflight")
        if k in raw and k != "enabled"
    }


def evaluate_arch_guardrails(
    model_type: str,
    arch_config: Any,
    param_count: int,
    estimated_memory_gb: float,
    guardrails: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Return advisory guardrail violation strings for diagnostics (profiling / pre-run).
    When arch_guardrails.enabled is false, returns no violations (diagnostics quiet).
    Training never uses this for skipping trials.
    """
    if not guardrails:
        return []
    if not bool(guardrails.get("enabled", False)):
        return []

    g = merge_arch_guardrails_for_type(guardrails, model_type)
    if not g:
        return []

    violations: List[str] = []

    max_params = g.get("max_params")
    max_memory = g.get("max_estimated_memory_gb")
    if max_params is not None and param_count > int(max_params):
        violations.append(f"params {param_count:,} > {int(max_params):,}")
    if max_memory is not None and estimated_memory_gb > float(max_memory):
        violations.append(f"est_memory_gb {estimated_memory_gb:.2f} > {float(max_memory):.2f}")

    if model_type == "CNN":
        cnn_preflight_cfg = g.get("cnn_preflight", {"enabled": True})
        if isinstance(cnn_preflight_cfg, dict) and bool(cnn_preflight_cfg.get("enabled", True)):
            custom_thresholds = {
                k: cnn_preflight_cfg[k]
                for k in DEFAULT_CNN_PREFLIGHT.keys()
                if k in cnn_preflight_cfg
            }
            cnn_violations = check_cnn_preflight(arch_config, custom_thresholds)
            violations.extend([f"cnn_preflight: {v}" for v in cnn_violations])

    return violations


def profile_model_space(
    exp_cfg,
    input_size: int,
    num_classes: int,
    batch_size: int,
    sample_input_shape: Optional[Tuple[int, ...]] = None,
    guardrails: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministically enumerate all arch seeds and estimate model size statistics.
    """
    if not exp_cfg.model_type_counts:
        raise RuntimeError("Model profiling requires build_search_space() before execution.")

    ordered_types = list(exp_cfg.model_types)
    counts = [exp_cfg.model_type_counts[mt] for mt in ordered_types]
    seed_ranges = []
    current_seed = 1
    for count in counts:
        seed_ranges.append((current_seed, current_seed + count - 1))
        current_seed += count

    sequence_length = _infer_sequence_length(sample_input_shape)
    per_type_records: Dict[str, List[Dict[str, Any]]] = {mt: [] for mt in ordered_types}
    heavy_overall: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []

    for model_type, (start_seed, end_seed) in zip(ordered_types, seed_ranges):
        generator_cls = model_generator_registry.get_generator(model_type)
        if generator_cls is None:
            logging.warning(f"No generator found for model type: {model_type}")
            continue

        for arch_seed in range(start_seed, end_seed + 1):
            set_seed(arch_seed)
            generator_instance = generator_cls(input_size, num_classes)
            model = generator_instance.generate_model()
            arch_cfg = getattr(model, "_arch_config", None)
            param_count = estimate_param_count(model)
            est_memory_gb = estimate_memory_gb(
                model_type=model_type,
                arch_config=arch_cfg,
                param_count=param_count,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )

            record = {
                "arch_seed": arch_seed,
                "param_count": param_count,
                "estimated_memory_gb": est_memory_gb,
                "arch_config": arch_cfg,
            }
            record["violations"] = evaluate_arch_guardrails(
                model_type=model_type,
                arch_config=arch_cfg,
                param_count=param_count,
                estimated_memory_gb=est_memory_gb,
                guardrails=guardrails,
            )
            per_type_records[model_type].append(record)
            heavy_overall.append({"model_type": model_type, **record})
            if record["violations"]:
                rejected_records.append({"model_type": model_type, **record})

    by_type_stats: Dict[str, Dict[str, Any]] = {}
    for model_type, records in per_type_records.items():
        if not records:
            continue
        params_sorted = sorted(records, key=lambda r: r["param_count"])
        memory_sorted = sorted(records, key=lambda r: r["estimated_memory_gb"])
        by_type_stats[model_type] = {
            "count": len(records),
            "min_params": params_sorted[0]["param_count"],
            "max_params": params_sorted[-1]["param_count"],
            "min_memory_gb": memory_sorted[0]["estimated_memory_gb"],
            "max_memory_gb": memory_sorted[-1]["estimated_memory_gb"],
            "max_param_seed": params_sorted[-1]["arch_seed"],
            "max_memory_seed": memory_sorted[-1]["arch_seed"],
            "top_heavy": memory_sorted[-3:][::-1],
        }

    heavy_overall_sorted = sorted(heavy_overall, key=lambda r: r["estimated_memory_gb"], reverse=True)
    rejected_by_type: Dict[str, int] = {}
    for item in rejected_records:
        rejected_by_type[item["model_type"]] = rejected_by_type.get(item["model_type"], 0) + 1

    return {
        "sequence_length": sequence_length,
        "by_type": by_type_stats,
        "overall_top_heavy": heavy_overall_sorted[:5],
        "rejected_total": len(rejected_records),
        "rejected_by_type": rejected_by_type,
        "rejected_examples": rejected_records[:5],
    }


def display_model_space_summary(dataset_name: str, model_profile: Dict[str, Any]) -> None:
    """Display deterministic model space size/memory summary."""
    print("\n" + "=" * 70)
    print("🧪 MODEL SPACE PROFILE (PRE-GENERATED)")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    seq_len = model_profile.get("sequence_length")
    if seq_len is not None:
        print(f"Inferred sequence length: {seq_len}")
    print("\nPer-model-type ranges:")
    print("-" * 70)
    by_type = model_profile.get("by_type", {})
    for model_type, stats in sorted(by_type.items()):
        print(
            f"  {model_type:12s} count={stats['count']:4d} | "
            f"params {stats['min_params']:,} .. {stats['max_params']:,} "
            f"(max seed {stats['max_param_seed']}) | "
            f"est_mem {stats['min_memory_gb']:.2f}GB .. {stats['max_memory_gb']:.2f}GB "
            f"(max seed {stats['max_memory_seed']})"
        )

    top_heavy = model_profile.get("overall_top_heavy", [])
    if top_heavy:
        print("\nMost memory-heavy predicted trials:")
        for item in top_heavy:
            print(
                f"  seed={item['arch_seed']:4d} type={item['model_type']:12s} "
                f"params={item['param_count']:,} est_mem={item['estimated_memory_gb']:.2f}GB"
            )

    rejected_total = int(model_profile.get("rejected_total", 0))
    if rejected_total > 0:
        print(f"\nGuardrail-rejected candidates (preflight): {rejected_total}")
        for model_type, count in sorted(model_profile.get("rejected_by_type", {}).items()):
            print(f"  {model_type:12s}: {count}")
    print("=" * 70)


def prompt_confirmation(dataset_name: str, total_trials: int) -> bool:
    """
    Prompt user to confirm before running experiments.
    
    Args:
        dataset_name: Name of the dataset
        total_trials: Total number of trials
    
    Returns:
        True if user confirms, False otherwise
    """
    print(f"\n⚠️  About to run {total_trials} trial(s) for dataset '{dataset_name}'")
    print("This may take a significant amount of time and resources.")
    
    while True:
        try:
            response = input("\nProceed with experiment? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Experiment cancelled by user")
            return False


def display_combined_trial_summary(
    dataset_summaries: List[Tuple[str, int, Dict[str, int]]],
    test_mode: bool = False
) -> None:
    """
    Display a combined summary of trial counts for multiple datasets.
    
    Args:
        dataset_summaries: List of (dataset_name, total_trials, model_type_breakdown) tuples
        test_mode: Whether running in test mode
    """
    print("\n" + "="*70)
    print("📊 COMBINED TRIAL COUNT SUMMARY")
    print("="*70)
    print(f"Mode: {'🧪 TEST' if test_mode else '🚀 PRODUCTION'}")
    
    total_all_trials = 0
    combined_model_type_breakdown = {}
    
    for dataset_name, total_trials, model_type_breakdown in dataset_summaries:
        total_all_trials += total_trials
        print(f"\nDataset: {dataset_name}")
        print(f"  Total Trials: {total_trials}")
        print("  Breakdown by Model Type:")
        for model_type, count in sorted(model_type_breakdown.items()):
            percentage = (count / total_trials * 100) if total_trials > 0 else 0
            print(f"    {model_type:20s}: {count:4d} trials ({percentage:5.1f}%)")
            # Accumulate for combined total
            combined_model_type_breakdown[model_type] = combined_model_type_breakdown.get(model_type, 0) + count
    
    print("\n" + "-" * 70)
    print(f"TOTAL ACROSS ALL DATASETS: {total_all_trials} trials")
    print("\nCombined Breakdown by Model Type:")
    for model_type, count in sorted(combined_model_type_breakdown.items()):
        percentage = (count / total_all_trials * 100) if total_all_trials > 0 else 0
        print(f"  {model_type:20s}: {count:4d} trials ({percentage:5.1f}%)")
    print("="*70)


def prompt_combined_confirmation(total_trials: int, num_datasets: int) -> bool:
    """
    Prompt user to confirm before running experiments for multiple datasets.
    
    Args:
        total_trials: Total number of trials across all datasets
        num_datasets: Number of datasets
    
    Returns:
        True if user confirms, False otherwise
    """
    print(f"\n⚠️  About to run {total_trials} total trial(s) across {num_datasets} dataset(s)")
    print("This may take a significant amount of time and resources.")
    
    while True:
        try:
            response = input("\nProceed with all experiments? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Experiments cancelled by user")
            return False


def diagnose_and_confirm(
    dataset_name: str,
    param_space: Dict[str, Any],
    exp_cfg,
    test_mode: bool = False,
    auto_confirm: bool = False,
    input_size: Optional[int] = None,
    num_classes: Optional[int] = None,
    batch_size: Optional[int] = None,
    sample_input_shape: Optional[Tuple[int, ...]] = None,
    guardrails: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Calculate trial counts, display summary, and prompt for confirmation.
    
    Args:
        dataset_name: Name of the dataset
        param_space: Ray Tune parameter space
        exp_cfg: ExperimentConfig instance
        test_mode: Whether running in test mode
        auto_confirm: If True, skip confirmation prompt (useful for automated runs)
    
    Returns:
        True if confirmed/auto-confirmed, False if cancelled
    """
    total_trials, model_type_breakdown = calculate_trial_counts(param_space, exp_cfg)
    
    display_trial_summary(dataset_name, total_trials, model_type_breakdown, test_mode)

    if input_size is not None and num_classes is not None and batch_size is not None:
        try:
            model_profile = profile_model_space(
                exp_cfg=exp_cfg,
                input_size=input_size,
                num_classes=num_classes,
                batch_size=batch_size,
                sample_input_shape=sample_input_shape,
                guardrails=guardrails,
            )
            display_model_space_summary(dataset_name, model_profile)
        except Exception as exc:
            logging.warning(f"Could not generate model-space diagnostic summary: {exc}")
    
    if auto_confirm:
        logging.info("🤖 Auto-confirm enabled, proceeding without user prompt")
        return True
    
    return prompt_confirmation(dataset_name, total_trials)
