# trainable.py
import os
import json
import gzip
import yaml
import torch 

import ray
from ray import tune
from ray.tune import get_context

from data.dataloader import create_dataloaders
from models.model_generator_registry import model_generator_registry
from train import evaluate_model, train_model
from utils.utils import set_seed, set_seed_deterministic, setup_file_logger
from utils.experiment_config import ExperimentConfig
from models.init_fns import INIT_FN_MAP 
from utils.trial_diagnostics import estimate_param_count, estimate_memory_gb


def _plain_python(obj):
    """Convert tuples to lists recursively so a config can be safe_dump'ed."""
    if isinstance(obj, dict):
        return {k: _plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain_python(v) for v in obj]
    return obj


def train_tune(
    config: ExperimentConfig,
    datasets_ref: ray.ObjectRef | None = None,
    *,
    simulate: bool = False,
):
    ctx = get_context()
    run_name = ctx.get_trial_name()  # e.g. "CNN_1e-3_bs64"
    trial_dir = ctx.get_trial_dir()
    
    # Create subdirectories
    logs_dir = os.path.join(trial_dir, "logs")
    ckpt_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # copy so we can extend it freely
    run_config = dict(config)  # Tune passes a FrozenDict
    run_config.update({
        "run_name": run_name,
        "run_dir": trial_dir,  # Use trial_dir as canonical run directory
        "logs_dir": logs_dir,
        "ckpt_dir": ckpt_dir
    })
    init_fn = INIT_FN_MAP[run_config["init_name"]]
    set_seed_deterministic(run_config["seed"])
    train_ds, val_ds, test_ds = ray.get(datasets_ref)
    train_loader = create_dataloaders(train_ds, run_config["batch_size"])
    val_loader = create_dataloaders(val_ds, run_config["batch_size"])
    test_loader = create_dataloaders(test_ds, run_config["batch_size"], shuffle=False)
    arch_seed = run_config["arch_seed"]
    init_seed  = run_config["init_seed"]
    
    if "_model_type_mapping" not in run_config:
        raise KeyError(
            "Trial config missing required '_model_type_mapping' (strict mode: only "
            "experiments.model_type_counts schedules are supported)."
        )
    mapping = run_config["_model_type_mapping"]
    ordered_types = mapping["ordered_types"]
    seed_ranges = mapping["seed_ranges"]
    model_type = None
    for mt, (start, end) in zip(ordered_types, seed_ranges):
        if start <= arch_seed <= end:
            model_type = mt
            break
    if model_type is None:
        raise ValueError(
            f"arch_seed={arch_seed} does not fall in any configured range "
            f"(ordered_types={ordered_types}, seed_ranges={seed_ranges})."
        )

    # Paper-era arch RNG (matches Sept / 654dd11 draws; not full torch deterministic mode).
    set_seed(arch_seed)

    generator_cls = model_generator_registry.get_generator(model_type)
    assert generator_cls
    generator_instance = generator_cls(train_ds.input_size, train_ds.num_classes)
    arch_model = generator_instance.generate_model()
    arch_cfg = arch_model._arch_config

    set_seed_deterministic(init_seed)
    model = generator_cls(train_ds.input_size, train_ds.num_classes, init_fn=init_fn).generate_model(
        fixed_arch_config=arch_cfg
    )

    # Guardrails: compute model-size signals before training starts.
    sample_shape = tuple(train_ds[0][0].shape) if len(train_ds) > 0 else None
    sequence_length = sample_shape[0] if sample_shape and len(sample_shape) >= 1 else None
    param_count = estimate_param_count(model)
    est_memory_gb = estimate_memory_gb(
        model_type=model_type,
        arch_config=arch_cfg,
        param_count=param_count,
        batch_size=int(run_config["batch_size"]),
        sequence_length=sequence_length,
    )
    # Keep init checkpoint in models subdir (for initial model state)
    model_dir = os.path.join(trial_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    init_ckpt = os.path.join(model_dir, f"{run_name}_init.pth.gz")

    if not os.path.exists(init_ckpt):  # write only once at initialization
        with gzip.open(init_ckpt, "wb") as fp:
            torch.save({"model_state_dict": model.state_dict()}, fp)

    arch_cfg = getattr(model, "_arch_config", None)
    if arch_cfg is not None:
        params_file = os.path.join(trial_dir, "params.json")
        try:
            with open(params_file, "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {}
        params["arch_config"] = arch_cfg
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

    # NOTE: reseeding with the original seed to maintain deterministic training behavior
    set_seed_deterministic(run_config["seed"])
    # ------------------------------------------------------------------
    # 4.  Set up per-trial logger folder
    # ------------------------------------------------------------------
    file_logger = setup_file_logger(run_name, logs_dir)  # Use logs_dir instead of run_dir
    file_logger.info(
        f"Running experiment {run_name} (simulate={simulate})..."
    )
    file_logger.info(f"Experiment Config: {run_config}")
    file_logger.info(f"Model Architecture ({model_type}): {model}")
    
    # Initialize summary data
    error = None
    train_metrics = None
    test_metrics = None
    
    if simulate:
        train_metrics = {
            "initial_val_f1_macro": 0.0,
            "best_val_f1_macro": 0.0,
            "train_time_sec": 0.0,
        }
        test_metrics = {
            "best_test_f1_macro": 0.0,
            "test_accuracy": 0.5,
        }
        file_logger.info(
            f"({run_name}) simulate mode: skipping train/eval; reporting placeholder metrics"
        )
        tune.report(
            {
                "initial_val_f1_macro": train_metrics["initial_val_f1_macro"],
                "best_val_f1_macro": train_metrics["best_val_f1_macro"],
                "best_test_f1_macro": test_metrics["best_test_f1_macro"],
                "test_accuracy": test_metrics["test_accuracy"],
                "train_time_sec": train_metrics["train_time_sec"],
            }
        )
    else:
        try:
            # ------------------------------------------------------------------
            # 5.  Train & evaluate using *your* utility funcs
            # ------------------------------------------------------------------
            train_result = train_model(
                model, train_loader, val_loader, run_config, train_ds.num_classes, file_logger
            )
            
            # Extract model and metrics from train_result dict
            # Handle both old format (just model) and new format (dict with model key)
            if not isinstance(train_result, dict):
                raise TypeError(
                    f"train_model must return a dict; got {type(train_result).__name__}"
                )
            best_model = train_result["model"]
            train_metrics = {
                "initial_val_f1_macro": train_result.get("initial_val_f1_macro"),
                "best_val_f1_macro": train_result.get("best_val_f1_macro"),
                "train_time_sec": train_result.get("train_time_sec"),
            }

            test_result = evaluate_model(
                best_model, test_loader, run_config, train_ds.num_classes, file_logger
            )
            
            # Extract test metrics from test_result dict
            test_metrics = {
                "best_test_f1_macro": test_result.get("f1_macro"),
                "test_accuracy": test_result.get("accuracy")
            }
            
            file_logger.info(f"({run_name}) Test Accuracy: {test_metrics['test_accuracy']:.4f}")
            
            # ------------------------------------------------------------------
            # 6.  Report back multiple metrics
            # ------------------------------------------------------------------
            tune.report({
                "initial_val_f1_macro": train_metrics["initial_val_f1_macro"],
                "best_val_f1_macro": train_metrics["best_val_f1_macro"],
                "best_test_f1_macro": test_metrics["best_test_f1_macro"],
                "test_accuracy": test_metrics["test_accuracy"],  # Required by Ray Tune config
                "train_time_sec": train_metrics["train_time_sec"]
            })
            
        except Exception as e:
            error = str(e)
            file_logger.error(f"Error during training: {error}")
            import traceback
            file_logger.error(traceback.format_exc())
            # Still report something to Ray
            tune.report({
                "initial_val_f1_macro": None,
                "best_val_f1_macro": None,
                "best_test_f1_macro": None,
                "train_time_sec": None,
                "error": error
            })
    
    # ------------------------------------------------------------------
    # 7.  Write summary.json and config_resolved.yaml
    # ------------------------------------------------------------------
    summary = {
        "experiment_id": run_config.get("experiment_id"),
        "exp_tag": run_config.get("exp_tag"),
        "exp_note": run_config.get("exp_note"),
        "git_commit": run_config.get("git_commit"),
        "dataset": run_config.get("dataset"),
        "model_type": model_type,
        "arch_seed": arch_seed,
        "model_param_count": param_count,
        "model_estimated_memory_gb": est_memory_gb,
        "init_name": run_config.get("init_name"),
        "init_seed": init_seed,
        "initial_val_f1_macro": train_metrics["initial_val_f1_macro"] if train_metrics else None,
        "best_val_f1_macro": train_metrics["best_val_f1_macro"] if train_metrics else None,
        "best_test_f1_macro": test_metrics["best_test_f1_macro"] if test_metrics else None,
        "train_time_sec": train_metrics["train_time_sec"] if train_metrics else None,
        "checkpoint_path": "checkpoints/model_best.pth.gz",
        "metrics_csv_path": f"logs/{run_name}_metrics.csv",
        "error": error
    }
    
    summary_file = os.path.join(trial_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    config_file = os.path.join(trial_dir, "config_resolved.yaml")
    with open(config_file, "w") as f:
        # safe_dump so the artifact stays loadable with yaml.safe_load. Tuples in
        # the config (e.g. _model_type_mapping.seed_ranges) would otherwise be
        # written as !!python/tuple tags that only the unsafe loader accepts.
        yaml.safe_dump(_plain_python(run_config), f, default_flow_style=False, sort_keys=False)
