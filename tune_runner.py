import os

os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = (
    "900"  # snapshot every seconds to avoid driver overload for longer-running runs
)
# Required by torch deterministic CUDA execution paths.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import logging
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import ray
from ray import tune
import torch

from data.dataset_registry import dataset_registry
from trainable import train_tune
from utils.config import load_config
from utils.experiment_config import ExperimentConfig  # updated path if needed
from utils.experiment_metadata import (
    generate_experiment_id,
    get_git_commit,
    get_collection_dir,
    create_manifest,
)
from utils.trial_diagnostics import diagnose_and_confirm
from utils.arch_config_extractor import extract_arch_configs_from_trials, save_arch_configs_df
from utils.utils import set_seed_deterministic


# -------------------------------
# Safe GPU concurrency calculations
# -------------------------------
def safe_trials_per_gpu(
    peak_reserved_gb: float,
    total_gpu_gb: float,
    safety_margin: float = 0.85,
) -> int:
    """
    Calculate safe number of concurrent trials per GPU based on memory usage.

    Args:
        peak_reserved_gb: Peak GPU memory reserved by a single trial (GB)
        total_gpu_gb: Total GPU memory available (GB)
        safety_margin: Safety margin (default 0.85 = 85% of available memory)

    Returns:
        Safe number of trials per GPU (minimum 1)
    """
    if peak_reserved_gb <= 0 or total_gpu_gb <= 0:
        return 1
    return max(1, int((total_gpu_gb * safety_margin) // peak_reserved_gb))


def measure_gpu_memory_probe(cfg: dict, dataset_name: str) -> float:
    """
    Run a quick GPU memory probe to measure peak memory usage.

    Uses experiments.model_types[0] and experiments.hyperparameters.batch_size.

    Args:
        cfg: Configuration dictionary (from load_config)
        dataset_name: Name of dataset to use

    Returns:
        Peak GPU memory reserved in GB, or None if measurement fails
    """
    if not torch.cuda.is_available():
        print("GPU memory probe: No CUDA available")
        return None

    try:
        # Load a small subset of data for probing
        dataset_class = dataset_registry.get_dataset(dataset_name)
        data_dir = cfg["data_dir"]
        rho, seed = cfg.get("rho"), cfg.get("seed")

        exp = cfg["experiments"]
        hp_bs = exp["hyperparameters"]["batch_size"]
        probe_batch_size = min(4, hp_bs)

        # Create small dataset (just a few samples for probe)
        train_dataset_full = dataset_class("train", data_dir, rho=rho, seed=seed)
        if len(train_dataset_full) > probe_batch_size * 2:  # Ensure at least 2 batches
            # Create subset for faster probing
            train_dataset = torch.utils.data.Subset(train_dataset_full, range(probe_batch_size * 2))
        else:
            train_dataset = train_dataset_full

        from data.dataloader import create_dataloaders
        train_loader = create_dataloaders(train_dataset, probe_batch_size, shuffle=False)

        from models.model_generator_registry import model_generator_registry
        from utils.utils import set_seed_deterministic

        model_type = exp["model_types"][0]
        set_seed_deterministic(42)  # Deterministic seed for probe

        model_generator = model_generator_registry.get_generator(model_type)
        sample_input_size = train_dataset_full.input_size
        sample_num_classes = train_dataset_full.num_classes

        model = model_generator(sample_input_size, sample_num_classes).generate_model()

        # Set up training components
        device = torch.device(cfg.get("device", "cuda"))
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Reset memory stats and run probe
        torch.cuda.reset_peak_memory_stats()

        # Run one training step
        model.train()
        batch_data, batch_labels = next(iter(train_loader))
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        # Measure peak memory
        torch.cuda.synchronize()
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

        return peak_reserved_gb

    except Exception as e:
        print(f"GPU memory probe failed: {e}")
        return None

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

_experiment_file_handler: Optional[logging.Handler] = None


def _setup_experiment_file_logging(collection_dir: Path) -> Path:
    """
    Append driver (tune_runner) logs under collection_dir next to MANIFEST / Ray storage.

    Replaces any file handler from a previous main() in the same process (e.g. simulate --all-datasets).
    """
    global _experiment_file_handler
    root = logging.getLogger()
    if _experiment_file_handler is not None:
        root.removeHandler(_experiment_file_handler)
        try:
            _experiment_file_handler.close()
        except Exception:
            pass
        _experiment_file_handler = None

    collection_dir = Path(collection_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)
    log_path = collection_dir / "tune_driver.log"
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    fh.setLevel(logging.INFO)
    root.addHandler(fh)
    _experiment_file_handler = fh
    logging.info("📝 Driver log (tail this file): %s", log_path)
    return log_path


@contextmanager
def ray_session_dir(session_dir: Path):
    """
    Context manager to temporarily change to a session directory for Ray initialization.
    Ray creates session directories in the current working directory, so we need
    to change to a subdirectory to keep the root clean.
    """
    original_cwd = os.getcwd()
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(str(session_dir))
        yield
    finally:
        os.chdir(original_cwd)


def main(
    cfg_path: str,
    test_mode: bool = False,
    dataset_name: str = None,
    reset: bool = False,
    exp_note: str = None,
    custom_params: dict = None,
    auto_confirm: bool = False,
    extract_arch_configs: bool = False,
    exp_id: str = None,
    restart_errored: bool = False,
    ray_address: str = None,
    simulate: bool = False,
):
    """
    Run Ray Tune experiment.

    Args:
        cfg_path: Path to unified config YAML
        test_mode: If True, store under experiments/test/ (does not change num_models or epochs)
        dataset_name: Name of dataset to run
        reset: Whether to reset/restart experiment
        exp_note: Optional experiment note
        custom_params: Optional explicit overrides (num_models, num_epochs); must stay consistent with model_type_counts
        auto_confirm: If True, skip confirmation prompt
        extract_arch_configs: If True, extract and return architecture configurations
        exp_id: Optional experiment ID (if provided, uses this instead of generating new one)
        restart_errored: When resuming (`Tuner.restore`), if True Ray re-queues trials that ended in ERROR
        ray_address: Address of an existing Ray cluster (e.g. "auto", "ray://head:10001",
            or "<head_ip>:6379"). If None, falls back to the RAY_ADDRESS env var; if that
            is also unset/empty, a local Ray instance is started (legacy single-machine path).
        simulate: If True, write under results/simulated-runs/, run the real trial setup
            (arch, params.json, init ckpt, summaries) but skip training and report placeholder
            metrics. Scheduling (GPU/CPU, concurrency) matches a normal run.

    Returns:
        dataset_name: Dataset name (required)
        reset: If True, start fresh and don't resume from existing checkpoints
    """
    if dataset_name is None:
        raise ValueError("dataset_name is required. Use: --config unified_config.yml --dataset <name>")
    
    cfg = load_config(cfg_path, dataset_name=dataset_name)

    # Resolve relative paths against project root
    from utils.path_utils import PROJECT_ROOT
    project_root = Path(PROJECT_ROOT)
    for key in ["save_dir", "data_dir"]:
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(project_root / cfg[key])
    if cfg.get("temp_dir") and cfg["temp_dir"] != "." and not Path(cfg["temp_dir"]).is_absolute():
        cfg["temp_dir"] = str(project_root / cfg["temp_dir"])

    if simulate:
        sim_root = project_root / "results" / "simulated-runs"
        sim_root.mkdir(parents=True, exist_ok=True)
        cfg["save_dir"] = str(sim_root)
        logging.info(f"🧪 Simulate mode: save_dir -> {cfg['save_dir']}")

    if "gpu_per_trial" not in cfg or cfg["gpu_per_trial"] is None:
        raise ValueError(
            "gpu_per_trial must be set under datasets.<name> in unified_config.yml"
        )
    gpu_per_trial_cfg = float(cfg["gpu_per_trial"])
    if gpu_per_trial_cfg <= 0:
        raise ValueError(f"gpu_per_trial must be > 0, got {gpu_per_trial_cfg!r}")

    # Apply custom parameter overrides
    if custom_params:
        if 'num_models' in custom_params:
            cfg["experiments"]["num_models"] = custom_params['num_models']
            logging.info(f"⚙️  Overriding num_models to {custom_params['num_models']}")
        if 'num_epochs' in custom_params:
            cfg["experiments"]["hyperparameters"]["num_epochs"] = custom_params['num_epochs']
            logging.info(f"⚙️  Overriding num_epochs to {custom_params['num_epochs']}")

    # Extract experiment metadata (optional)
    exp_tag = cfg.get("exp_tag")
    if exp_note is None:
        exp_note = cfg.get("exp_note")

    # Generate experiment ID and setup collection directory
    # Use provided exp_id if available (for grouping multiple datasets in same collection)
    # Otherwise generate a new one
    if exp_id is None:
        exp_id = generate_experiment_id(test_mode=test_mode)
    git_commit = get_git_commit()
    base_results_dir = Path(cfg["save_dir"])
    collection_dir = get_collection_dir(base_results_dir, exp_id, exp_tag, test_mode)
    dataset_name = cfg["experiments"]["dataset"]
    # Ray Tune stores trials at storage_path/<name>/<trial_name>
    # We use collection_dir as storage_path and dataset_name as name
    # So trials end up at: collection_dir/<dataset_name>/<trial_name>
    dataset_experiment_dir = collection_dir / dataset_name

    _setup_experiment_file_logging(collection_dir)
    
    # Create MANIFEST.yaml at collection level (only if doesn't exist)
    create_manifest(
        collection_dir,
        exp_id,
        exp_tag,
        exp_note,
        git_commit,
        test_mode,
        config_path=cfg_path
    )
    
    logging.info(f"📁 Collection directory: {collection_dir}")
    logging.info(f"📊 Dataset experiment directory: {dataset_experiment_dir}")
    
    set_seed_deterministic(cfg["seed"])  # set seed for search algorithms/schedulers
    # refer to the docs for details on proper seeding: https://docs.ray.io/en/latest/tune/faq.html

    ray_tmp = str(cfg["temp_dir"]).strip()
    ray_tmp = str(Path(ray_tmp).expanduser())
    Path(ray_tmp).mkdir(parents=True, exist_ok=True)
    logging.info(f"Ray session root (_temp_dir): {ray_tmp}")
    
    # Shutdown Ray if it's already initialized (e.g., from a previous experiment)
    if ray.is_initialized():
        logging.info("🔄 Ray already initialized, shutting down first...")
        ray.shutdown()
    
    runtime_env = {
        "env_vars": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        # Ship the project to remote workers. On a multi-node cluster this is how
        # imports like `data.dataset_registry`, `trainable`, etc. become available.
        # If you've baked the repo into the Docker image AND it lives at the same
        # absolute path on every node, you can disable this by setting
        # MGT_RAY_SHIP_CODE=0 to avoid re-zipping the project at job start.
        "working_dir": str(project_root),
    }
    if os.environ.get("MGT_RAY_SHIP_CODE", "1") == "0":
        runtime_env.pop("working_dir", None)
        logging.info(
            "🚫 MGT_RAY_SHIP_CODE=0: not shipping working_dir to Ray workers; "
            "relying on the project being present at the same path on every node."
        )

    ray.init(
        include_dashboard=False,
        runtime_env=runtime_env,
        _temp_dir=ray_tmp,
    )

    # --- Pre-load dataset once and keep a reference in the object store ----
    # dataset_name already extracted above
    dataset_class = dataset_registry.get_dataset(dataset_name)
    data_dir = cfg["data_dir"]
    rho, seed = cfg.get("rho"), cfg.get("seed")
    if rho and seed:
        logging.warning(
            f"rho and seed set to {rho} and {seed}! this will result in an imbalanced dataset. Please ensure that this was intentional"
        )
    train_dataset = dataset_class("train", data_dir, rho=rho, seed=seed)
    val_dataset = dataset_class("val", data_dir)
    test_dataset = dataset_class("test", data_dir)
    # Preload the datasets into Ray's object store
    preloaded_data_refs = []
    preloaded_data_refs.append(ray.put(train_dataset))
    preloaded_data_refs.append(ray.put(val_dataset))
    preloaded_data_refs.append(ray.put(test_dataset))

    logging.info(f"Type of preloaded_data_ref: {type(preloaded_data_refs)}")
    logging.info(f"Preloaded data reference: {preloaded_data_refs}")

    exp_cfg = ExperimentConfig(cfg["experiments"], cfg["seed"], test_mode=test_mode)
    param_space = exp_cfg.build_search_space(
        input_size=train_dataset.input_size,
        num_classes=train_dataset.num_classes,
    )
    
    sample_input_shape = tuple(train_dataset[0][0].shape) if len(train_dataset) > 0 else None
    batch_size = int(exp_cfg.hyperparameters["batch_size"])

    # Diagnostic check: calculate and display trial counts, prompt for confirmation
    if not diagnose_and_confirm(
        dataset_name,
        param_space,
        exp_cfg,
        test_mode=test_mode,
        auto_confirm=auto_confirm,
        input_size=train_dataset.input_size,
        num_classes=train_dataset.num_classes,
        batch_size=batch_size,
        sample_input_shape=sample_input_shape,
        guardrails=cfg["experiments"].get("arch_guardrails", {}),
    ):
        logging.info("❌ Experiment cancelled by user")
        return
    
    param_space.update(
        {
            "save_dir": str(dataset_experiment_dir),  # Use collection/dataset directory
            "device": cfg["device"],
            "dataset": cfg["experiments"]["dataset"],
            "seed": cfg["seed"],
            # Pass experiment metadata to trainable
            "experiment_id": exp_id,
            "exp_tag": exp_tag,
            "exp_note": exp_note,
            "git_commit": git_commit,
        }
    )

    # Detect GPUs; per-trial GPU count from unified_config (gpu_per_trial), optional per-dataset override.
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        num_gpus = 0

    if num_gpus <= 0:
        raise RuntimeError(
            "No CUDA GPUs visible to this process; tune_runner requires at least one GPU "
            "(torch.cuda.is_available() and device_count() > 0)."
        )

    total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logging.info(f"🎮 Found {num_gpus} GPU(s), total memory: {total_gpu_gb:.1f}GB per GPU")

    gpu_per_trial = gpu_per_trial_cfg
    logging.info(
        f"🎯 GPU scheduling: gpu_per_trial={gpu_per_trial} (from config); "
        "concurrency follows Ray/cluster resources (no max_concurrent_trials cap). "
        "Trial priority follows arch_seed grid order (TRANSFORMER → RNN → CNN, CNN largest-first)."
    )
    
    trainable = tune.with_resources(
        tune.with_parameters(
            train_tune,
            datasets_ref=preloaded_data_refs,
            simulate=simulate,
        ),
        resources={"cpu": 3, "gpu": gpu_per_trial},
    )

    # Use dataset name as experiment name for Ray Tune
    experiment_name = dataset_name
    experiment_dir = dataset_experiment_dir

    # arch_seed grid is globally ordered for scheduling priority (FIFO queue: TR → RNN → CNN, ascending arch_seed per block).
    # No max_concurrent_trials cap — many trials can run in parallel; Ray still prefers earlier/higher-priority trials.
    tune_cfg = tune.TuneConfig(metric="test_accuracy", mode="max")

    try:
        if experiment_dir.exists() and not reset:
            logging.info(
                f"🔁 Resuming Ray Tune experiment from {experiment_dir} "
                f"(retry_errored={restart_errored})"
            )
            tuner = tune.Tuner.restore(
                str(experiment_dir),
                trainable=trainable,
                param_space=param_space,
                resume_unfinished=True,
                restart_errored=restart_errored,
            )
        else:
            # Import datetime here so it's available for manifest updates later
            from datetime import datetime

            if reset and experiment_dir.exists():
                logging.info(f"🔄 RESET MODE: Starting fresh experiment (existing directory will be backed up)")
                # Backup existing directory by renaming it with timestamp
                backup_dir = experiment_dir.parent / f"{experiment_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                experiment_dir.rename(backup_dir)
                logging.info(f"📦 Backed up existing experiment to: {backup_dir}")

            logging.info(f"🚀 Starting new Ray Tune experiment at {experiment_dir}")
            # Create dataset experiment directory and ensure collection_dir exists for Ray storage
            collection_dir.mkdir(parents=True, exist_ok=True)
            experiment_dir.mkdir(parents=True, exist_ok=True)

            tuner = tune.Tuner(
                trainable,
                param_space=param_space,
                tune_config=tune_cfg,
                run_config=tune.RunConfig(
                    name=experiment_name,
                    storage_path=str(collection_dir),  # Ray storage at collection level: collection/<dataset_name>/<trials>
                    log_to_file=False,  # Log to stdout instead of file
                ),
            )

        # Run the experiment - all output goes to stdout
        results = tuner.fit()

        # Log best results
        best = results.get_best_result("test_accuracy", "max")
        best_result_msg = f"🏆 Best config: {best.config} \n test_accuracy: {best.metrics['test_accuracy']}"
        logging.info(best_result_msg)

        # Extract architecture configurations if requested
        arch_configs_df = None
        # Automatically extract and save architecture configurations to collection root
        if not simulate:
            try:
                logging.info("📊 Extracting architecture configurations from all trials...")
                arch_configs_df = extract_arch_configs_from_trials(
                    experiment_dir,
                    include_trial_metadata=True
                )

                if arch_configs_df is not None and len(arch_configs_df) > 0:
                    # Save to parquet file in the collection root directory
                    arch_configs_path = collection_dir / "arch_configs.parquet"
                    save_arch_configs_df(arch_configs_df, arch_configs_path, format="parquet")
                    logging.info(f"✅ Saved {len(arch_configs_df)} architecture configurations to {arch_configs_path}")
                else:
                    logging.warning("⚠️  No architecture configurations found to extract")
            except Exception as e:
                logging.warning(f"⚠️  Could not extract architecture configurations: {e}")
        else:
            logging.info("🧪 Simulate: skipping arch_configs.parquet extraction")

        # Update MANIFEST with runtime statistics
        try:
            from utils.experiment_metadata import update_collection_manifest

            # Count total trials and get date range
            num_trials = len(results)
            if num_trials > 0:
                # Get timestamps from results if available
                trial_times = []
                for result in results:
                    if hasattr(result, 'metrics') and 'timestamp' in result.metrics:
                        trial_times.append(result.metrics['timestamp'])
                    elif hasattr(result, 'start_time'):
                        trial_times.append(result.start_time)

                # Update manifest with runtime stats
                # Read existing manifest to append dataset instead of overwriting
                manifest_path = collection_dir / "MANIFEST.yaml"
                existing_datasets = []
                if manifest_path.exists():
                    try:
                        import yaml
                        with open(manifest_path, 'r') as f:
                            existing_manifest = yaml.safe_load(f) or {}
                            existing_datasets = existing_manifest.get("datasets", [])
                    except Exception:
                        pass
                
                # Append this dataset if not already in list
                if dataset_name not in existing_datasets:
                    existing_datasets.append(dataset_name)
                
                manifest_updates = {
                    "datasets": existing_datasets,  # List of all datasets in this collection
                    "total_models": num_trials,
                    "completed_at": datetime.now().isoformat(),
                }

                if trial_times:
                    manifest_updates["date_range_start"] = min(trial_times)
                    manifest_updates["date_range_end"] = max(trial_times)

                update_collection_manifest(collection_dir, **manifest_updates)
                logging.info("📊 Updated collection manifest with runtime statistics")
        except Exception as e:
            logging.warning(f"Could not update manifest with runtime stats: {e}")
        
        # Return arch_configs_df if extracted (for programmatic access)
        if extract_arch_configs:
            return arch_configs_df
    finally:
        # Always shutdown Ray to prevent reinit errors when running multiple experiments
        if ray.is_initialized():
            logging.info("🛑 Shutting down Ray...")
            ray.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Path to unified config file (configs/experiments/unified_config.yml)"
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (required)"
    )
    ap.add_argument(
        "--extract-arch-configs",
        action="store_true",
        help="Extract and save architecture configurations DataFrame to arch_configs.parquet"
    )
    ap.add_argument(
        "--retry-errored",
        action="store_true",
        help="On resume only: re-run trials that finished with ERROR (no effect on fresh runs)",
    )
    ap.add_argument(
        "--exp-id",
        default=None,
        metavar="ID",
        help="Existing experiment ID (see collection MANIFEST.yaml); resume Tune in that folder",
    )
    ap.add_argument(
        "--ray-address",
        default=None,
        metavar="ADDR",
        help=(
            "Address of an existing Ray cluster to attach to (e.g. 'auto', "
            "'ray://head:10001', '<head_ip>:6379'). Falls back to the RAY_ADDRESS "
            "env var; if neither is set, a local Ray instance is started."
        ),
    )
    args = ap.parse_args()
    main(
        args.config,
        dataset_name=args.dataset,
        extract_arch_configs=args.extract_arch_configs,
        exp_id=args.exp_id,
        restart_errored=args.retry_errored,
        ray_address=args.ray_address,
    )
