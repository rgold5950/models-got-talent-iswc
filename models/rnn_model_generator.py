import random

import numpy as np
import torch.nn as nn
from .base_model_generator import BaseModelGenerator

class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.1,
                 init_fn=nn.init.xavier_normal_):
        super().__init__()
        self.rnn     = nn.LSTM(input_size, hidden_size, num_layers=1,
                               batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        def _weights_init(m):
            if isinstance(m, (nn.LSTM, nn.GRU)):
                for n, p in m.named_parameters():
                    if "weight" in n:
                        init_fn(p)
                    elif "bias" in n:
                        p.data.zero_()
            elif isinstance(m, nn.Linear):
                init_fn(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.dropout(x)
    
class Encoder(nn.Module):
    def __init__(self, rnn_configs, init_fn):
        super().__init__()
        self.rnn_layers = nn.ModuleList(
            [RNNLayer(**cfg, init_fn=init_fn) for cfg in rnn_configs]
        )

    def forward(self, x):
        for layer in self.rnn_layers:
            x = layer(x)
        return x
    
class RNNClassifier(nn.Module):
    def __init__(self, num_classes, rnn_configs, init_fn): 
        super().__init__()
        self.backbone = Encoder(rnn_configs, init_fn)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_configs[-1]["hidden_size"], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self.classifier.apply(
            lambda m: init_fn(m.weight) if isinstance(m, nn.Linear) else None
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = x[:, -1, :]
        return self.classifier(x)
    
class RNNModelGenerator(BaseModelGenerator):
    def __init__(self, input_size, num_classes, init_fn=nn.init.xavier_normal_):
        self.input_size = input_size
        self.num_classes = num_classes
        self.init_fn     = init_fn  

    def sample_arch_config(self):
        param_ranges = {
            "hidden_sizes": np.arange(8, 512, 8).tolist(),
            "num_layers"  : [2, 3, 4],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        num_layers = random.choice(param_ranges["num_layers"])
        configs    = []
        in_size    = self.input_size
        for i in range(num_layers):
            hidden  = random.choice(param_ranges["hidden_sizes"])
            drop    = random.choice(param_ranges["dropout_rate"]) if i < num_layers-1 else 0.0
            configs.append(
                {"input_size": in_size, "hidden_size": hidden, "dropout_rate": drop}
            )
            in_size = hidden
        return configs
    
    def generate_model(self, fixed_arch_config=None):
        
        # # Choose either LSTM or GRU for entire model
        # rnn_type = random.choice(param_ranges['rnn_type'])
        # num_layers = random.choice(param_ranges['num_layers'])
        
        # # Generate hidden sizes and dropout rates for each layer
        # hidden_sizes = [random.choice(param_ranges['hidden_sizes']) for _ in range(num_layers)]
        # dropout_rates = [random.choice(param_ranges['dropout_rate']) for _ in range(num_layers-1)] + [0.0]
        
        # Generate hidden sizes and dropout rates for each layer
        

        rnn_configs = fixed_arch_config or self.sample_arch_config()
        model = RNNClassifier(self.num_classes, rnn_configs, init_fn=self.init_fn)
        model._arch_config = rnn_configs  # expose for logging / freezing
        return model
