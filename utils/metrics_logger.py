import os
import csv
from typing import Any
class MetricsLogger:
    BASE_HEADER = [  
        "epoch",
        "phase",           # "initial_train", "train", "val", "test"
        "accuracy",
        "loss",
        "f1_macro",
        "f1_weighted",
        "max_perm_accuracy",
        "mean_perm_accuracy",
        "max_perm_f1_macro",
        "mean_perm_f1_macro",
        "max_perm_f1_weighted",
        "mean_perm_f1_weighted",
        # best-so-far (same order as above for the first block)
        "best_accuracy",
        "best_f1_macro",
        "best_f1_weighted",
        "best_max_perm_accuracy",
        "best_mean_perm_accuracy",
        "best_max_perm_f1_macro",
        "best_mean_perm_f1_macro",
        "best_max_perm_f1_weighted",
        "best_mean_perm_f1_weighted",
    ]
    PHASES = ("train", "val", "test", "initial_train", "initial_val")
    
    def _empty_best(self) -> dict[str, Any]:
        """Return dict with every metric set to None (shape taken from BASE_HEADER)."""
        base = {}
        for col in self.BASE_HEADER:
            if col.startswith("best_"):
                base[col.removeprefix("best_")] = None
        return base


    def __init__(self, log_dir, experiment_id):
        """
        Args:
        log_dir (str): Directory in which to save logs.
        experiment_id (str): A unique name for the experiment (used for the filename).
        resume_epoch (int, optional): If training is resuming,
                                        rows with epoch >= resume_epoch are discarded.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"{experiment_id}_metrics.csv")
        # dynamic header + in-memory row buffer ------------------------  # <<< NEW
        self.header: list[str] = list(MetricsLogger.BASE_HEADER)    # <<< NEW
        self.rows:  list[dict[str, float | int | None]] = []             # <<< NEW
        self.best_by_phase: dict[str, dict[str, Any]] = {
            p: self._empty_best() for p in self.PHASES
        }

        self.best_metrics = None
        self.start_epoch = 0

        if os.path.exists(self.log_file):
            with open(self.log_file, newline="") as f:
                reader = csv.DictReader(f)
                # preserve column order from file
                if reader.fieldnames:
                    for col in reader.fieldnames:
                        if col not in self.header:
                            self.header.append(col)
                self.rows.extend(reader)

            if self.rows:
                self.start_epoch = int(self.rows[-1]["epoch"]) + 1
                val_rows = [r for r in self.rows if r["phase"] == "val"]
                if val_rows:
                    self.best_metrics = {
                        k.removeprefix("best_"): float(v)
                        for k, v in val_rows[-1].items()
                        if k.startswith("best_") and v not in ("", None, "None")
                    }

        self._write_csv()   

    # ------------------------------------------------------------------
    # helper: write current header+rows to disk
    # ------------------------------------------------------------------
    def _write_csv(self) -> None:
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
            
    def _parse_best_metrics(self, row: dict[str, str | float | None]) -> dict[str, float]:
        return {
            k.replace("best_", ""): float(v)
            for k, v in row.items()
            if k.startswith("best_") and v not in (None, "", "None")
        }
      
    def check_phase_complete(self, phase: str, last_expected_epoch: int) -> bool:
        """
        Check if the run is complete by comparing the epoch of the test phase
        with the expected last epoch.
        Args:
        last_epoch (int): The last epoch of the training run.
        Returns:
        bool: True if the run is complete, False otherwise.
        """
        phase_rows = [r for r in self.rows if r["phase"] == phase]
        if not phase_rows:
            return False
        last_epoch = int(phase_rows[-1]["epoch"])
        return (last_epoch + 1) >= last_expected_epoch

    def resume(self, resume_epoch: int | None = None) -> None:
        """
        Reset internal state so that training can resume.
        If the run is empty (no CSV or no rows) simply set start_epoch
        and return without touching best_metrics.
        """
        # ------------------------------------------------------------------
        # 1. Fast-exit if the log file does not exist OR has no data rows
        # ------------------------------------------------------------------
        if not os.path.exists(self.log_file):
            # Nothing to resume from – treat as brand-new run
            self.start_epoch = resume_epoch or 0
            self.best_metrics = None
            return

        # ------------------------------------------------------------------
        # 2. Load existing rows as dicts (header handled automatically)
        # ------------------------------------------------------------------
        with open(self.log_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            data_rows = list(reader)                 # ← dicts, not lists

        if not data_rows:
            self.start_epoch = resume_epoch or 0
            self.best_metrics = None
            return

        # ------------------------------------------------------------------
        # 3. If no explicit epoch is supplied, resume after last logged epoch
        # ------------------------------------------------------------------
        if resume_epoch is None:
            last_logged_epoch = int(data_rows[-1]["epoch"])
            self.start_epoch = last_logged_epoch + 1
            return

        # ------------------------------------------------------------------
        # 4. Filter out any rows **on/after** resume_epoch
        # ------------------------------------------------------------------
        filtered_rows = [
            r for r in data_rows if int(r["epoch"]) < resume_epoch
        ]

        # overwrite CSV with preserved column order
        with open(self.log_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(filtered_rows)

        self.start_epoch = resume_epoch

        # ------------------------------------------------------------------
        # 5. Re-compute best_metrics from newest *val* row (if any)
        # ------------------------------------------------------------------
        val_rows = [r for r in filtered_rows if r["phase"] == "val"]
        self.best_metrics = (
            self._parse_best_metrics(val_rows[-1]) if val_rows else None
        )

    def log(self, epoch: int, phase: str, metrics: dict[str, int | None]):
        # 1. update best-of-run on validation rows --------------------- #
        if phase == "val":
            if self.best_metrics is None:
                self.best_metrics = metrics.copy()
            else:
                for k, v in metrics.items():
                    if v is not None and (self.best_metrics.get(k) is None or v > self.best_metrics[k]):
                        self.best_metrics[k] = v
        if self.best_metrics is None:
            self.best_metrics = {k: None for k in metrics}

        # 2. assemble row --------------------------------------------- #
        row: dict[str, int | None] = {
            "epoch": epoch,
            "phase": phase,
            **metrics,
            **{f"best_{k}": v for k, v in self.best_metrics.items()},
        }

        # 3. extend header in place (preserve order) ------------------- #
        for col in row.keys():
            if col not in self.header:
                self.header.append(col)

        self.rows.append(row)
        self._write_csv()

    def get_best_metrics(self):
        """Return the best metrics dictionary."""
        return self.best_metrics

    def get_start_epoch(self):
        """Return the epoch that training should resume from."""
        return self.start_epoch
    
    def get_last_epoch(self) -> int | None:
        return int(self.rows[-1]["epoch"]) if self.rows else None
    
    def get_last_phase_metric(self, phase: str, metric: str) -> float | None:
        rows = [r for r in self.rows if r["phase"] == phase]
        if not rows:
            return None
        try:
            return float(rows[-1][metric])
        except (KeyError, TypeError, ValueError):
            return None