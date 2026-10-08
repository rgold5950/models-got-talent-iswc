import joblib
import numpy as np
import os
import torch
from time import time
from torch.utils.data import Dataset


from utils.sliding_window import sliding_window


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)

class MyoGymDataset(Dataset):
    
    # Ensure that the dataset path is set correctly
    config = {
        # Data loading parameters
        'window': 100,
        'overlap': 50,
        'input_size': 3,

        # Dataset parameters
        'dataset': 'myogym',
        'data_file': 'myogym_3_sr_50.joblib',
        'num_classes': 31,

    }
    
    def __init__(self, phase, root_dir: str, rho: float = 1.0, seed: int = None):
        if rho and rho != 1.0:
            raise ValueError("exponential class weighting not supported for this dataset")
        self.num_classes = self.config['num_classes']
        self.input_size = self.config['input_size']
        self.filename = os.path.join(root_dir, self.config['data_file'])

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
        unique = np.unique(self.labels)
        print(f"[{phase}] unique labels: {unique.tolist()}")
        assert self.config['input_size'] == self.data_raw[phase]['data'].shape[1]
        assert unique.max() == self.num_classes - 1, (
            f"Highest label is {unique.max()}, but num_classes is "
            f"{self.num_classes}. Did you forget to subtract 1 or "
            f"set num_classes correctly?"
        )


        # print('The dataset is: {}. The phase is: {}. The size of the dataset '
        #       'is: {}'.format(self.config['dataset'], phase, self.data.shape))

    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        # print('Data loading completed in {:.0f}m {:.0f}s'
        #       .format(time_elapsed // 60, time_elapsed % 60))

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]
        data = torch.from_numpy(data)

        label = torch.from_numpy(np.asarray(self.labels[index]))
        return data, label

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS

    root_dir = DATASET_CONFIGS["myogym"]['data_root']
    phase = "train"  # could also be "val" or "test"

    dataset = MyoGymDataset(phase=phase, root_dir=root_dir, rho=1.0)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Print the first batch
    for batch_x, batch_y in dataloader:
        print("Batch X shape:", batch_x.shape)  # [batch_size, window, input_size]
        print("Batch Y shape:", batch_y.shape)  # [batch_size]
        break