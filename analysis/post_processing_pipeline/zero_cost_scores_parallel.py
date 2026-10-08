"""
Compute zero-cost metrics for every trained model in each dataset specified in the configs.  
Batch-sizes come from a pre-existing full_results_pkl.pkl.
Results are saved alongside the original pickle.

Columns in the final df:
    run_dir | model_path | batch_size | measure1 | measure2 | etc.
"""
import gzip
import random
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from foresight.pruners import predictive  # pip install zero-cost-nas
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import ray
from contextlib import nullcontext, contextmanager
# Silence PyTorch transformer nested tensor warnings
warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)

from utils.path_utils import setup_path
setup_path()
from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS
from data.dataset_registry import dataset_registry
from models.model_generator_registry import model_generator_registry


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        print("No CUDA available")


def extract_arch_seed_from_run_dir(run_dir: str) -> Optional[int]:
    """
    Extract arch_seed from run_dir name.
    
    Example: 'train_tune_4e3bb_01554_1554_arch_seed=1555,...' -> 1555
    
    Args:
        run_dir: Directory name containing arch_seed parameter
        
    Returns:
        arch_seed as integer, or None if not found
    """
    match = re.search(r'arch_seed=(\d+)', str(run_dir))
    if match:
        return int(match.group(1))
    return None

model_patterns = ("*.pt", "*.pth", "*.pth.gz", "*.init.pth.gz")
load_state_dict = True  # checkpoints are state_dicts
NOISE_LEVELS = [
    0.0, 
    # 0.01,
    # 0.05, 
    # 0.1
]   

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def discover_models(root: Path, patterns: tuple[str, ...]) -> pd.DataFrame:
    """Return df with columns run_dir | model_path (first file that matches)."""
    rows: List[dict] = []
    for run_dir in sorted(root.iterdir()):
        models_dir = run_dir / "models"
        if not models_dir.is_dir():
            continue
        for pat in patterns:
            matches = sorted(models_dir.glob(pat))
            if matches:
                rows.append({"run_dir": run_dir.name, "model_path": matches[0]})
                break
    if not rows:
        raise RuntimeError(f"No checkpoints found under {root}")
    return pd.DataFrame(rows)


def build_model(
    num_classes: int, input_size: int, arch: str = "cnn"
) -> torch.nn.Module:
    generator = model_generator_registry.get_generator(arch.upper())(
        input_size, num_classes
    )
    return generator.generate_model()


def load_checkpoint(
    path: Path, num_classes: int, input_size: int, arch="cnn"
) -> tuple[torch.nn.Module, float]:
    t0 = time.perf_counter()
    handle = gzip.open(path, "rb") if path.suffix == ".gz" else path
    checkpoint = torch.load(handle, map_location=device)

    model = build_model(num_classes, input_size, arch=arch)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model = model.to(device)
    return model, time.perf_counter() - t0

def is_transformer_like(model: torch.nn.Module) -> bool:
    attention_types = (
        torch.nn.MultiheadAttention,
        torch.nn.TransformerEncoderLayer,
        torch.nn.TransformerDecoderLayer,
        torch.nn.TransformerEncoder,
        torch.nn.TransformerDecoder,
        torch.nn.Transformer,
    )

    return any(isinstance(m, attention_types) for m in model.modules())

@contextmanager
def sdpa_math_only():
    # Only relevant on CUDA; on CPU this is effectively a no-op.
    if torch.cuda.is_available():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            yield
    else:
        yield

def maybe_sdpa_math_only(model: torch.nn.Module):
    # we only want to turn off fast math if absolutely necessary (e.g. for transformer models)
    return sdpa_math_only() if is_transformer_like(model) else nullcontext()

def compute_metrics(
    model: torch.nn.Module,
    batch_size: int,
    dataset,
    num_classes: int,
    sigma: float = 0.0,           # default to no noise
) -> tuple[Dict[str, float], dict]:
    """Returns (metrics_dict, timing_dict)"""
    t_data0 = time.perf_counter()
    g = torch.Generator()
    g.manual_seed(123)  # ensure same shuffle each time
    if sigma > 0:
        # noise collate_fn
        def noise_collate_fn(batch):
            Xs, ys = zip(*batch)
            X = torch.stack(Xs).to(device)
            noise = torch.randn_like(X) * sigma
            return X + noise, torch.stack(ys).to(device)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=noise_collate_fn,
        )
    else:
        # no noise
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
    t_data = time.perf_counter() - t_data0

    # 2) zero-cost measures
    t_meas0 = time.perf_counter()
    with maybe_sdpa_math_only(model):
        measures = predictive.find_measures(
            model,
            dataloader=loader,
            dataload_info=("random", 1, num_classes),
            device=device,
        )
    t_meas = time.perf_counter() - t_meas0

    metrics = {k: float(v) for k, v in measures.items()}
    
    # 3) Calculate total FLOPs for the model
    try:
        # Try to import FLOPs counting libraries
        try:
            from thop import profile, clever_format
            use_thop = True
        except ImportError:
            try:
                from fvcore.nn import FlopCountMode, flop_count
                use_fvcore = True
            except ImportError:
                use_thop = False
                use_fvcore = False
        
        # Get input size from dataset
        if hasattr(dataset, 'config') and 'input_size' in dataset.config:
            input_size = dataset.config['input_size']
        elif hasattr(dataset, 'input_size'):
            input_size = dataset.input_size
        else:
            # Fallback: get from first batch
            sample_batch = next(iter(loader))
            if isinstance(sample_batch, (list, tuple)):
                sample_input = sample_batch[0]
            else:
                sample_input = sample_batch
            input_size = sample_input.shape[1:]  # Skip batch dimension
        
        # Create dummy input with actual batch_size
        if isinstance(input_size, (list, tuple)):
            dummy_input = torch.randn(batch_size, *input_size).to(device)
        else:
            # Assume 1D input
            dummy_input = torch.randn(batch_size, input_size).to(device)
        
        model.eval()
        with torch.no_grad():
            if use_thop:
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                metrics["total_flops"] = float(flops)
            elif use_fvcore:
                flop_dict, _ = flop_count(model, (dummy_input,))
                total_flops = sum(flop_dict.values())
                metrics["total_flops"] = float(total_flops)
            else:
                # Fallback: if no FLOPs library available, set to NaN
                metrics["total_flops"] = float("nan")
    except Exception as e:
        # If FLOPs calculation fails, set to NaN but don't fail the whole computation
        metrics["total_flops"] = float("nan")
    
    timing = {
        "data_sec":    t_data,
        "measures_sec": t_meas,
        "total_sec":    time.perf_counter() - t_meas0,
    }
    return metrics, timing

@ray.remote(num_gpus=0.25 if torch.cuda.is_available() else 0)
def run_model_eval(
    row_dict: dict,
    dataset_ref,
    num_classes: int,
    input_size: int,
    sigma: float = 0.0,
) -> dict:    # Silence PyTorch warnings in Ray workers
    warnings.filterwarnings(
        "ignore",
        message=r"enable_nested_tensor is True, but self.use_nested_tensor is False.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Using a non-full backward hook when the forward contains multiple autograd Nodes.*",
        category=UserWarning,
    )
    set_seed(int(row_dict["arch_seed"]))
    model, load_sec = load_checkpoint(Path(row_dict["model_path"]), num_classes, input_size, row_dict["model_type"])
    set_seed(42)

    metrics, tdict = compute_metrics(
        model,
        int(row_dict["batch_size"]),
        dataset_ref,
        num_classes,
        sigma=sigma,           # ← pass noise level through
    )

    tqdm.write(
        f"✔ {row_dict['run_dir']:<35} "
        f"(bs={row_dict['batch_size']:3d}) load={load_sec:5.2f}s  "
        f"data={tdict['data_sec']:5.2f}s  meas={tdict['measures_sec']:5.2f}s  "
        f"total={tdict['total_sec']:5.2f}s  "
        f"synflow={metrics['synflow']:.3e}  "
        f"jacov={metrics.get('jacob_cov', float('nan')):.3e}"
    )

    return {
        **row_dict,
        "noise_sigma": sigma,
        "load_sec": load_sec,
        **tdict,
        **metrics,
    }


def run_for_dataset(dataset_name: str, config_override: dict = None):
    """
    Run zero-cost score computation for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        config_override: Optional config dict to override DATASET_CONFIGS (for test mode)
    """
    # Use override config if provided, otherwise use DATASET_CONFIGS
    if config_override and dataset_name in config_override:
        cfg = config_override[dataset_name]
    elif dataset_name not in DATASET_CONFIGS:
        print(f"⚠️  No config for dataset '{dataset_name}', skipping.")
        return
    else:
        cfg = DATASET_CONFIGS[dataset_name]
    print(f"\n=== Zero-cost scores for dataset: {dataset_name} ===")

    try:
        full_df = pd.read_pickle(cfg["full_results_pkl"])
    except FileNotFoundError:
        print(f"❌ Skipping {dataset_name}: full_runs_df.pkl not found.")
        return

    if "batch_size" not in full_df.columns:
        raise ValueError(f"'batch_size' missing in {cfg['full_results_pkl']}")

    # Log initial state
    print(f"📊 Loaded {len(full_df)} rows from full_results.pkl")
    print(f"   Unique run_dir values: {full_df['run_dir'].nunique()}")

    try:
        # Convert results_dir string to Path if needed
        results_dir = Path(cfg["results_dir"]) if isinstance(cfg["results_dir"], str) else cfg["results_dir"]
        run_df = discover_models(results_dir, model_patterns)
        print(f"📂 Found {len(run_df)} model checkpoints in filesystem")
    except RuntimeError as e:
        print(f"❌ Skipping {dataset_name}: {e}")
        return

    # Merge dataframes - don't use arch_seed from full_df as it may be incorrect
    # We'll extract it from run_dir name instead
    models = full_df[["batch_size","run_dir","model_type"]] \
             .merge(run_df, on="run_dir", how="inner")
    
    print(f"🔗 After merging: {len(models)} models matched")
    
    # Drop duplicates by run_dir (keep first occurrence) - duplicates can come from merge
    before_dedup = len(models)
    models = models.drop_duplicates(subset=['run_dir'], keep='first')
    if len(models) < before_dedup:
        duplicates_dropped = before_dedup - len(models)
        print(f"⚠️  Dropped {duplicates_dropped} duplicate run_dir entries after merge")
    
    # Extract arch_seed from run_dir name (source of truth)
    models["arch_seed"] = models["run_dir"].apply(extract_arch_seed_from_run_dir)
    
    # Check if any arch_seed extraction failed
    missing_arch_seed = models["arch_seed"].isna().sum()
    if missing_arch_seed > 0:
        print(f"⚠️  Warning: Could not extract arch_seed for {missing_arch_seed} models")
    
    print(f"✅ Will process {len(models)} unique models")

    if "_" in dataset_name:
        base_name, rho_str = dataset_name.split("_", 1)
        rho = float(rho_str)
        seed = cfg.get('seed', 42)
    else:
        base_name = dataset_name
        rho = 1.0
        seed = None

    seed = cfg.get("seed", 42)   # or whatever default you prefer
    DatasetCls = dataset_registry.get_dataset(base_name)
    dataset    = DatasetCls(
        phase="train",
        root_dir=cfg["data_root"],
        rho=rho,      # pass the imbalance parameter
        seed=seed,    # must supply a seed when rho != 1.0 | None
    )
    num_classes = getattr(dataset, "num_classes", len(set(dataset.labels)))
    input_size  = dataset.config["input_size"]
    dataset_ref = ray.put(dataset)

    # ─── loop over all noise levels ────────────────────────────────────
    for sigma in NOISE_LEVELS:
        label = f"σ={sigma:.3f}" if sigma>0 else "clean"
        print(f"\n--- Running {label} ---")

        # fire off one batch of ray tasks per model
        futures = [
            run_model_eval.remote(
                row.to_dict(),
                dataset_ref,
                num_classes,
                input_size,
                sigma=sigma,             # ← pass sigma here
            )
            for _, row in models.iterrows()
        ]

        # Track progress while waiting for results
        total_models = len(futures)
        print(f"   Processing {total_models} models...")
        completed = 0
        
        # Use ray.wait to show progress
        remaining = futures.copy()
        records = []
        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=1, timeout=1.0)
            if ready:
                completed += len(ready)
                records.extend(ray.get(ready))
                if completed % max(1, total_models // 10) == 0 or completed == total_models:
                    print(f"   Progress: {completed}/{total_models} models completed ({100*completed/total_models:.1f}%)")
        
        # Get any remaining results
        if remaining:
            records.extend(ray.get(remaining))
            completed = len(records)
            print(f"   Progress: {completed}/{total_models} models completed (100%)")

        # assemble and save
        scores_df = pd.DataFrame.from_records(records)
        # choose filename: overwrite clean, suffix others
        base = Path(cfg["zero_cost_pkl"])
        if sigma == 0.0:
            out_pkl = base
        else:
            out_pkl = base.with_name(f"{base.stem}_noise{sigma:.3f}.pkl")

        scores_df.to_pickle(out_pkl)
        print(f"💾 Saved {len(scores_df)} rows → {out_pkl}")


def main():
    ray.init(ignore_reinit_error=True)
    for dataset_name in DATASET_CONFIGS:
        run_for_dataset(dataset_name)
    ray.shutdown()


if __name__ == "__main__":
    main()
