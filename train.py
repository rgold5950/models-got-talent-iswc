# train.py
import gzip
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.metrics_logger import MetricsLogger
from utils.utils import compute_metrics


# -------------------------------
# GPU memory probe helper
# -------------------------------
def log_gpu_peak(prefix: str = ""):
    """Log current GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated() / 1024**3
    reserv = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        f"{prefix} GPU total={total:.2f}GB | "
        f"peak_alloc={alloc:.2f}GB | "
        f"peak_reserved={reserv:.2f}GB"
    )


def reset_gpu_peak_memory():
    """Reset GPU peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _save_half_gzip(model_state_dict: dict, path: str, epoch: int):
    """
    Store weights in FP16 and write them through gzip.
    """
    half_state = {k: v.half() for k, v in model_state_dict.items()}
    checkpoint = {"epoch": epoch, "model_state_dict": half_state}
    with gzip.open(path, "wb") as f:  # torch.save works with file-like obj
        torch.save(checkpoint, f)


def _load_half_gzip(path: str, device: torch.device):
    """
    Load a FP16+gzip checkpoint, cast back to FP32.
    """
    with gzip.open(path, "rb") as f:
        ckpt = torch.load(f, map_location=device)
    float_state = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
    return ckpt["epoch"], float_state


def evaluate_model_on_loader(model, dataloader, config, num_classes):
    device = torch.device(config["device"])
    model = model.to(device)
    model.eval()
    actual_labels = []
    predicted_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            actual_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    # compute_metrics is assumed to return a tuple of 9 values.
    return compute_metrics(actual_labels, predicted_labels, num_classes)

def unpack_class_counts(cls_cnts: dict[str, int]) -> dict[str, int]:
    cls_cnt_formatted = {}
    for cls, cnt in cls_cnts.items():
        cls_cnt_formatted[f"class_{cls}"] = cnt
    return cls_cnt_formatted

def train_model(model, train_loader, val_loader, config, num_classes, file_logger):
    # Use logs_dir for metrics CSV (run_dir is trial_dir, logs_dir is trial_dir/logs)
    log_dir = config.get("logs_dir", config["run_dir"])
    metric_logger = MetricsLogger(log_dir, config["run_name"])

    skip_run = metric_logger.check_phase_complete("train", config["num_epochs"])
    # Use ckpt_dir from config (created by trainable)
    ckpt_dir = config["ckpt_dir"]
    model_save_path = os.path.join(ckpt_dir, "model_best.pth.gz")

    device = torch.device(config["device"])
    train_start_time = time.time()

    if skip_run:
        file_logger.info(
            f"Run {config['run_name']} already completed. Skipping training."
        )
        assert os.path.exists(model_save_path), "Checkpoint file not found."

        # --- load the FP16-+-gzip checkpoint and cast back to FP32 ---
        _, state = _load_half_gzip(model_save_path, device)
        model.load_state_dict(state)
        # Extract metrics from logger for return
        best_metrics = metric_logger.get_best_metrics()
        # Try to get initial val f1_macro from logger (from initial_val phase)
        initial_val_rows = [r for r in metric_logger.rows if r.get("phase") == "initial_val"]
        initial_val_f1_macro = float(initial_val_rows[0]["f1_macro"]) if initial_val_rows else None
        return {
            "model": model,
            "initial_val_f1_macro": initial_val_f1_macro,
            "best_val_f1_macro": best_metrics.get("f1_macro") if best_metrics else None,
            "train_time_sec": 0.0  # Already completed
        }

    # Run one initial inference loop to log untrained metrics.
    initial_train = evaluate_model_on_loader(model, train_loader, config, num_classes)
    initial_val = evaluate_model_on_loader(model, val_loader, config, num_classes)

    # Log the initial metrics (logged with epoch 0).
    init_train_metrics = {
        "accuracy": initial_train[0],
        "f1_weighted": initial_train[1],
        "f1_macro": initial_train[2],
        "max_perm_accuracy": initial_train[3],
        "mean_perm_accuracy": initial_train[4],
        "max_perm_f1_macro": initial_train[7],
        "mean_perm_f1_macro": initial_train[8],
        "max_perm_f1_weighted": initial_train[5],
        "mean_perm_f1_weighted": initial_train[6],
    }
    init_train_metrics.update(unpack_class_counts(initial_train[9]))
    init_val_metrics = {
        "accuracy": initial_val[0],
        "f1_weighted": initial_val[1],
        "f1_macro": initial_val[2],
        "max_perm_accuracy": initial_val[3],
        "mean_perm_accuracy": initial_val[4],
        "max_perm_f1_macro": initial_val[7],
        "mean_perm_f1_macro": initial_val[8],
        "max_perm_f1_weighted": initial_val[5],
        "mean_perm_f1_weighted": initial_val[6],
    }
    init_train_metrics.update(unpack_class_counts(initial_val[9]))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    # GPU memory probe: measure peak memory usage after first training step
    gpu_memory_probe_completed = False
    peak_reserved_gb = None

    resume_epoch = 0
    initial_val_f1_macro = init_val_metrics["f1_macro"]

    if os.path.exists(model_save_path):
        resume_epoch, state = _load_half_gzip(model_save_path, device)
        model.load_state_dict(state)
        metric_logger.resume(resume_epoch)
        file_logger.info(
            f"Found checkpoint at epoch {resume_epoch}. Resuming training..."
        )
    else:
        # ckpt_dir is already created by trainable, no need to makedirs
        metric_logger.log(0, "initial_train", init_train_metrics)
        metric_logger.log(0, "initial_val", init_val_metrics)

    model = model.to(device)

    # Use the logger's recorded starting epoch.
    start_epoch = metric_logger.get_start_epoch()
    
    for current_epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        running_loss = 0.0
        train_actual, train_predicted = [], []
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # GPU memory probe: measure after first training step
            if not gpu_memory_probe_completed and current_epoch == 1 and batch_idx == 0:
                reset_gpu_peak_memory()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Complete GPU memory probe after first step
            if not gpu_memory_probe_completed and current_epoch == 1 and batch_idx == 0:
                log_gpu_peak(prefix="[mem-probe]")
                if torch.cuda.is_available():
                    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
                    print(f"[mem-probe] Measured peak reserved: {peak_reserved_gb:.2f}GB")
                gpu_memory_probe_completed = True

            running_loss += loss.item() * inputs.size(0)
            train_actual.extend(labels.cpu().numpy())
            train_predicted.extend(predicted.cpu().numpy())
        scheduler.step()

        # Calculate training metrics.
        train_results = compute_metrics(train_actual, train_predicted, num_classes)
        train_metrics = {
            "accuracy": train_results[0],
            "loss": running_loss / len(train_loader.dataset),
            "f1_weighted": train_results[1],
            "f1_macro": train_results[2],
            "max_perm_accuracy": train_results[3],
            "mean_perm_accuracy": train_results[4],
            "max_perm_f1_weighted": train_results[5],
            "mean_perm_f1_weighted": train_results[6],
            "max_perm_f1_macro": train_results[7],
            "mean_perm_f1_macro": train_results[8],
        }
        train_metrics.update(unpack_class_counts(train_results[9]))     
        metric_logger.log(current_epoch, "train", train_metrics)

        # Validation loop.
        model.eval()
        val_running_loss = 0.0
        val_actual, val_predicted = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_actual.extend(labels.cpu().numpy())
                val_predicted.extend(predicted.cpu().numpy())
        val_results = compute_metrics(val_actual, val_predicted, num_classes)
        val_metrics = {
            "accuracy": val_results[0],
            "loss": val_running_loss / len(val_loader.dataset),
            "f1_weighted": val_results[1],
            "f1_macro": val_results[2],
            "max_perm_accuracy": val_results[3],
            "mean_perm_accuracy": val_results[4],
            "max_perm_f1_weighted": val_results[5],
            "mean_perm_f1_weighted": val_results[6],
            "max_perm_f1_macro": val_results[7],
            "mean_perm_f1_macro": val_results[8],
        }
        val_metrics.update(unpack_class_counts(val_results[9]))     
        metric_logger.log(current_epoch, "train", train_metrics)
        # Before logging new validation metrics, store the current best f1_macro.
        prev_best = (
            metric_logger.get_best_metrics()["f1_macro"]
            if metric_logger.get_best_metrics()["f1_macro"]
            else -1.0
        )
        metric_logger.log(current_epoch, "val", val_metrics)
        new_best = metric_logger.get_best_metrics()["f1_macro"]

        # Save checkpoint when a new best validation f1_macro is achieved.
        if new_best > prev_best:
            _save_half_gzip(model.state_dict(), model_save_path, current_epoch)
            file_logger.info(
                f"Saved new checkpoint at epoch {current_epoch} "
                f"(val f1_macro = {new_best:.4f})"
            )

        file_logger.info(
            f"Epoch {current_epoch}: Train Loss: {train_metrics['loss']:.4f}, Train F1 Macro: {train_metrics['f1_macro']}, Val Loss: {val_metrics['loss']:.4f}, Val F1 Macro: {val_metrics['f1_macro']:.4f}"
        )

    # After training, load the best model.
    if os.path.exists(model_save_path):
        _, state = _load_half_gzip(model_save_path, torch.device(config["device"]))
        model.load_state_dict(state)
    
    train_time_sec = time.time() - train_start_time
    best_metrics = metric_logger.get_best_metrics()
    
    torch.cuda.empty_cache()
    
    # Return model and metrics dict
    return {
        "model": model,
        "initial_val_f1_macro": initial_val_f1_macro,
        "best_val_f1_macro": best_metrics.get("f1_macro") if best_metrics else None,
        "train_time_sec": train_time_sec,
        "peak_gpu_memory_gb": peak_reserved_gb
    }


def evaluate_model(model, test_loader, config, num_classes, file_logger):
    # Use logs_dir for metrics CSV
    log_dir = config.get("logs_dir", config["run_dir"])
    metric_logger = MetricsLogger(log_dir, config["run_name"])

    skip_run = metric_logger.check_phase_complete("test", config["num_epochs"])
    if skip_run:
        file_logger.info(
            f"Run {config['run_name']} already completed. Skipping evaluation."
        )
        # Return dict with test metrics from logger
        test_metrics_dict = {}
        test_accuracy = metric_logger.get_last_phase_metric("test", "accuracy")
        test_f1_macro = metric_logger.get_last_phase_metric("test", "f1_macro")
        return {
            "accuracy": test_accuracy,
            "f1_macro": test_f1_macro
        }

    test_results = evaluate_model_on_loader(model, test_loader, config, num_classes)
    test_metrics = {
        "accuracy": test_results[0],
        "f1_weighted": test_results[1],
        "f1_macro": test_results[2],
        "max_perm_accuracy": test_results[3],
        "mean_perm_accuracy": test_results[4],
        "max_perm_f1_weighted": test_results[5],
        "mean_perm_f1_weighted": test_results[6],
        "max_perm_f1_macro": test_results[7],
        "mean_perm_f1_macro": test_results[8],
    }
    test_metrics.update(unpack_class_counts(test_results[9]))     
        

    epoch = metric_logger.get_last_epoch()
    assert epoch is not None, "Epoch not found in metrics logger."
    metric_logger.log(epoch, "test", test_metrics)

    # Return dict with key metrics
    return {
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"]
    }
