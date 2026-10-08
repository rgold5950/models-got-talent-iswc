from torch.utils.data import Dataset

from .hhar_dataset import HHARDataset
from .motionsense_dataset import MotionSenseDataset
from .pamap2_dataset import PAMAP2Dataset
from .random_dataset import RandomDataset
from .rwhar_dataset import RealWorldDataset
from .mobiact_dataset import MobiactV2Dataset
from .myogym_dataset import MyoGymDataset

class DatasetRegistry:
    def __init__(self):
        self.datasets = {}
    
    def register(self, name: str, dataset_class: Dataset):
        self.datasets[name] = dataset_class

    def get_dataset(self, name: str) -> Dataset:
        return self.datasets.get(name)

dataset_registry = DatasetRegistry()
dataset_registry.register('motionsense', MotionSenseDataset)
dataset_registry.register('pamap2', PAMAP2Dataset)
dataset_registry.register('hhar', HHARDataset)
dataset_registry.register('rwhar', RealWorldDataset)
dataset_registry.register('myogym', MyoGymDataset)
dataset_registry.register('random', RandomDataset)
dataset_registry.register("mobiactv2", MobiactV2Dataset)

