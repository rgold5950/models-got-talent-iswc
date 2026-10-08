from torch.utils.data import DataLoader, Dataset
from data.dataset_registry import dataset_registry
from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS

def create_dataloaders(
    dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 2
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

def create_dataset_from_name(name: str, phase: str) -> Dataset:
    DatasetCls = dataset_registry.get_dataset(name)
    dataset    = DatasetCls(
        phase=phase,
        root_dir=DATASET_CONFIGS[name]["data_root"],
    )
    return dataset