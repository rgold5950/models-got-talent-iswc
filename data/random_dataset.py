# random_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

from .motionsense_dataset import opp_sliding_window  # reuse the same fn


class RandomDataset(Dataset):
    """
    Synthetic replacement for MotionSenseDataset that generates
    random windows + random labels on the fly.

    It is meant to be **API-compatible** with MotionSenseDataset:
        ds = RandomDataset(phase="train", root_dir=".", n_windows=25_000)

    Parameters
    ----------
    phase : str
        Ignored (kept only for signature parity).
    root_dir : str
        Ignored (kept only for signature parity).
    n_windows : int, optional
        How many sliding windows you want **after** segmentation.
    seed : int or None, optional
        Set for deterministic pseudo-data.
    """

    config = {
        "window": 100,
        "overlap": 50,
        "input_size": 3,
        "dataset": "random",
        "data_file": None,
        "num_classes": 6,
    }

    def __init__(
        self,
        phase,
        root_dir: str,
        n_windows: int = 10_000,
        rho: float = 1.0,
        seed: int = None,
    ):
        if rho and rho != 1.0:
            raise ValueError(
                "exponential class weighting not supported for this dataset"
            )
        super().__init__()
        self.num_classes = self.config["num_classes"]
        self.input_size = self.config["input_size"]
        self.window = self.config["window"]
        self.overlap = self.config["overlap"]

        rng = np.random.default_rng(seed)
        stride = self.overlap
        raw_len = (n_windows - 1) * stride + self.window

        raw_data = rng.standard_normal(  # (T, C)
            size=(raw_len, self.input_size), dtype=np.float32
        )
        raw_labels = rng.integers(
            low=0, high=self.num_classes, size=raw_len, dtype=np.uint8
        )

        self.data, self.labels = opp_sliding_window(
            raw_data, raw_labels, self.window, self.overlap
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index])
        y = torch.from_numpy(np.asarray(self.labels[index])) 
        return x, y
