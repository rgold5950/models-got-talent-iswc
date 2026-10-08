"""
CNN random architecture matching commit 654dd11 / Sept 2025 paper runs.

- Encoder ends with full-length max_pool1d (not adaptive_avg_pool1d).
- Sampling: out_channels in [8, 1024] step 8, num_layers in [1..7], kernels 2-9.
- After set_seed(arch_seed), generate_model() matches
  the old trainable draw order (no unique-cache or preflight constraints).
"""

import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .base_model_generator import BaseModelGenerator


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        dropout_rate=0.1,
        init_fn=nn.init.xavier_normal_,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        def _weights_init(m):
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.GRU, nn.LSTM)):
                init_fn(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        x = self.dropout(self.relu(self.conv(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, conv_configs, init_fn=nn.init.xavier_normal_):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [ConvBlock(**cfg, init_fn=init_fn) for cfg in conv_configs]
        )

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        return x.squeeze(2)


class Classifier(nn.Module):
    def __init__(self, num_classes, conv_configs, init_fn=nn.init.xavier_normal_):
        super().__init__()
        self.backbone = Encoder(conv_configs, init_fn=init_fn)
        self.softmax = nn.Sequential(
            nn.Linear(conv_configs[-1]["out_channels"], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, inputs):
        backbone = self.backbone(inputs)
        return self.softmax(backbone)


class CNNModelGenerator(BaseModelGenerator):
    """Paper-era CNN random architecture (use with utils.set_seed for arch draws)."""

    def __init__(self, input_size, num_classes, init_fn=nn.init.xavier_normal_):
        self.input_size = input_size
        self.num_classes = num_classes
        self.init_fn = init_fn

    def create_conv_configs(self, input_size, num_layers, param_ranges):
        configs = []
        in_channels = input_size
        for _ in range(num_layers):
            out_channels = random.choice(param_ranges["out_channels"])
            kernel_size = random.choice(param_ranges["kernel_size"])
            stride = random.choice(param_ranges["stride"])
            dropout_rate = random.choice(param_ranges["dropout_rate"])
            configs.append(
                {
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "dropout_rate": dropout_rate,
                }
            )
            in_channels = out_channels
        return configs

    def generate_model(self, fixed_arch_config=None):
        if fixed_arch_config is not None:
            conv_layers = fixed_arch_config
        else:
            param_ranges = {
                "out_channels": np.arange(8, 1025, 8).tolist(),
                "kernel_size": [2, 3, 4, 5, 6, 7, 8, 9],
                "stride": [1],
                "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
                "num_layers": [1, 2, 3, 4, 5, 6, 7],
            }
            num_layers = random.choice(param_ranges["num_layers"])
            conv_layers = self.create_conv_configs(self.input_size, num_layers, param_ranges)

        model = Classifier(self.num_classes, conv_layers, init_fn=self.init_fn)
        model._arch_config = conv_layers
        return model
