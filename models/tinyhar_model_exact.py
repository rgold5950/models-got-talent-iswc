# models/tinyhar_reference.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from models.base_model_generator import BaseModelGenerator


def _nhead_for_F(F: int) -> int:
    # Tiny + safe: pick the largest <=4 that divides F
    for d in (4, 3, 2, 1):
        if F % d == 0:
            return d
    return 1


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.v(torch.tanh(self.W(h)))  # (B,T,1)
        w = torch.softmax(scores, dim=1)
        return (w * h).sum(dim=1)               # (B,H)


class TinyHARReference(nn.Module):
    """
    Fixed “reference” TinyHAR:
      - 4 grouped Conv1d layers per channel, k=5, stride=2, BN+ReLU
      - 1 transformer encoder layer across channels
      - fusion bottleneck F_star = 2F
      - 1-layer LSTM with hidden size = F_star
      - temporal attention + learnable gamma
      - classifier: Linear(512)->ReLU->Linear(K) is *not* in TinyHAR; TinyHAR uses lightweight head.
        We keep a single Linear to stay faithful & lightweight.
    """
    def __init__(
        self,
        n_channels: int,
        num_classes: int,
        F: int = 8,
        conv_layers: int = 4,
        kernel_size: int = 5,
        stride: int = 2,
        transformer_layers: int = 1,
        transformer_ff_mult: int = 2,
        transformer_dropout: float = 0.1,
        init_fn=nn.init.xavier_normal_,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.F = F
        self.F_star = 2 * F

        conv_out = n_channels * F
        conv: list[nn.Module] = []

        conv.append(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=conv_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=n_channels,
                bias=False,
            )
        )
        conv.append(nn.BatchNorm1d(conv_out))
        conv.append(nn.ReLU(inplace=True))

        for _ in range(conv_layers - 1):
            conv.append(
                nn.Conv1d(
                    in_channels=conv_out,
                    out_channels=conv_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=n_channels,
                    bias=False,
                )
            )
            conv.append(nn.BatchNorm1d(conv_out))
            conv.append(nn.ReLU(inplace=True))

        self.individual_conv = nn.Sequential(*conv)

        nhead = _nhead_for_F(F)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=F,
            nhead=nhead,
            dim_feedforward=transformer_ff_mult * F,
            dropout=transformer_dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.fusion_fc = nn.Linear(n_channels * F, self.F_star)

        self.lstm = nn.LSTM(
            input_size=self.F_star,
            hidden_size=self.F_star,   # match common TinyHAR setup
            num_layers=1,
            batch_first=True,
        )

        self.attn = TemporalAttention(self.F_star)
        self.gamma = nn.Parameter(torch.zeros(()))

        self.classifier = nn.Linear(self.F_star, num_classes)

        # init
        self.apply(lambda m: init_fn(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        B, T, C = x.shape
        if C != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got {C}")

        x = x.transpose(1, 2)  # (B,C,T)
        x = self.individual_conv(x)  # (B,C*F,T*)
        B, CF, Tstar = x.shape

        x = x.view(B, self.n_channels, self.F, Tstar).permute(0, 3, 1, 2).contiguous()  # (B,T*,C,F)

        xt = x.view(B * Tstar, self.n_channels, self.F)  # (B*T*,C,F)
        xt = self.channel_transformer(xt)
        x = xt.view(B, Tstar, self.n_channels, self.F)

        x = x.reshape(B, Tstar, self.n_channels * self.F)
        x = self.fusion_fc(x)  # (B,T*,F*)

        h, _ = self.lstm(x)  # (B,T*,F*)
        c = self.attn(h)     # (B,F*)
        last = h[:, -1, :]   # (B,F*)
        z = last + self.gamma * c

        return self.classifier(z)


class TinyHARReferenceModelGenerator(BaseModelGenerator):
    """
    Fixed benchmark generator: always returns the same TinyHAR reference config.
    """
    def __init__(self, input_size: int, num_classes: int, init_fn=nn.init.xavier_normal_):
        self.input_size = input_size
        self.num_classes = num_classes
        self.init_fn = init_fn

        # expose fixed config for logging
        self.fixed_arch_config: dict[str, Any] = {
            "F": 8,
            "conv_layers": 4,
            "kernel_size": 5,
            "stride": 2,
            "transformer_layers": 1,
            "transformer_ff_mult": 2,
            "transformer_dropout": 0.1,
            "fusion_ratio": 2.0,  # informational
            "lstm_hidden": 16,    # informational == F_star
        }

    def sample_arch_config(self) -> dict[str, Any]:
        # fixed
        return dict(self.fixed_arch_config)

    def generate_model(self, fixed_arch_config: dict[str, Any] | None = None) -> nn.Module:
        cfg = fixed_arch_config or self.fixed_arch_config
        model = TinyHARReference(
            n_channels=self.input_size,
            num_classes=self.num_classes,
            F=int(cfg["F"]),
            conv_layers=int(cfg["conv_layers"]),
            kernel_size=int(cfg["kernel_size"]),
            stride=int(cfg["stride"]),
            transformer_layers=int(cfg["transformer_layers"]),
            transformer_ff_mult=int(cfg["transformer_ff_mult"]),
            transformer_dropout=float(cfg["transformer_dropout"]),
            init_fn=self.init_fn,
        )
        model._arch_config = dict(cfg)
        return model
