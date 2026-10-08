import joblib
import numpy as np
import os
import torch
from time import time
from torch.utils.data import Dataset


from utils.sliding_window import sliding_window
from utils.exponential_subsample import _exponential_subsample

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)

class MotionSenseDataset(Dataset):
    
    # Ensure that the dataset path is set correctly
    config = {
        # Data loading parameters
        'window': 100,
        'overlap': 50,
        'input_size': 3,

        # Dataset parameters
        'dataset': 'motionsense',
        'data_file': 'motionsense.pkl',
        'num_classes': 6,

    }
    
    def __init__(self, phase, root_dir: str, rho: float = 1.0, seed: int | None = None, noise_std: float | None = None):
        self.num_classes = self.config['num_classes']
        self.input_size = self.config['input_size']
        self.filename = os.path.join(root_dir, self.config['data_file'])
        self.noise_std = noise_std

        # If the prepared dataset doesn't exist, give a message and exit
        if not os.path.isfile(self.filename):
            print(f'The data is not available at {self.filename}. '
                  'Ensure that the data is present in the directory.')
            exit(0)

        # Loading the data
        self.data_raw = self.load_dataset(self.filename)
        # Obtaining the segmented data
        self.data, self.labels = \
            opp_sliding_window(self.data_raw[phase]['data'],
                               self.data_raw[phase]['labels'],
                               self.config['window'],
                               self.config['overlap'])
        
        if rho and rho != 1.0 and phase == 'train': # NOTE: this should only run for training data
            assert seed is not None, "seed cannot be none!"
            cnts = np.unique(self.labels, return_counts=True)[-1]
            print(f'[BEFORE] exponential imbalancing -> min label count: {cnts.min()} max label count: {cnts.max()} ratio min/max: {cnts.min()/cnts.max()}')
            self.data, self.labels = _exponential_subsample(
                self.data, self.labels, rho, seed
            )
            cnts = np.unique(self.labels, return_counts=True)[-1]
            print(f'[AFTER] exponential imbalancing with rho ({rho}) -> min label count: {cnts.min()} max label count: {cnts.max()} ratio min/max: {cnts.min()/cnts.max()}')


        unique = np.unique(self.labels)
        print(f"[{phase}] unique labels: {unique.tolist()}")
        assert self.config['input_size'] == self.data_raw[phase]['data'].shape[1]
        assert unique.max() == self.num_classes - 1, (
            f"Highest label is {unique.max()}, but num_classes is "
            f"{self.num_classes}. Did you forget to subtract 1 or "
            f"set num_classes correctly?"
        )
        print('The dataset is: {}. The phase is: {}. The size of the dataset '
              'is: {}'.format(self.config['dataset'], phase, self.data.shape))

    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        print('Data loading completed in {:.0f}m {:.0f}s'
              .format(time_elapsed // 60, time_elapsed % 60))

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index, :, :]
        x = torch.from_numpy(x)

        y = torch.from_numpy(np.asarray(self.labels[index]))
        if self.noise_std is not None and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y

if __name__ == "__main__":
    from collections import Counter

    from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS

    phase = 'test'  # or 'train', 'val'
    root_dir = DATASET_CONFIGS["motionsense"]["data_root"]

    # Load the dataset
    dataset = MotionSenseDataset(phase=phase, root_dir=root_dir)

    # Count label frequencies
    label_counts = Counter(dataset.labels)

    # Print results
    print(f"[{phase}]Number of samples per class:")
    for class_id in range(dataset.num_classes):
        print(f"Class {class_id}: {label_counts.get(class_id, 0)} samples")