# models/tinyhar_model_generator.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.base_model_generator import BaseModelGenerator


def _largest_divisor_leq(n: int, k: int) -> int:
    for d in range(min(k, n), 0, -1):
        if n % d == 0:
            return d
    return 1


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H)
        scores = self.v(torch.tanh(self.W(h)))   # (B, T, 1)
        w = torch.softmax(scores, dim=1)         # (B, T, 1)
        c = (w * h).sum(dim=1)                   # (B, H)
        return c


class TinyHARBackbone(nn.Module):
    """
    TinyHAR-style backbone:
      - grouped Conv1d per channel -> (B, C*F, T*)
      - reshape to (B, T*, C, F)
      - transformer encoder across channels at each time step
      - fusion bottleneck linear: (C*F) -> F_star
      - 1-layer LSTM: (B, T*, F_star) -> (B, T*, H)
      - temporal attention + learnable gamma
    """
    def __init__(
        self,
        n_channels: int,
        F: int,
        conv_layers: int,
        kernel_size: int,
        stride: int,
        transformer_layers: int,
        transformer_nhead: int,
        transformer_ff_mult: int,
        transformer_dropout: float,
        fusion_ratio: float,          # F_star = round(fusion_ratio * F)
        lstm_hidden: int,
        lstm_dropout: float,
        init_fn=nn.init.xavier_normal_,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.F = F

        # conv out channels = C*F; grouped by C
        conv_out = n_channels * F
        conv: list[nn.Module] = []

        # 1st: (B, C, T) -> (B, C*F, T*)
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

        # remaining: keep same shape, still grouped by C
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

        # transformer across channels: sequence len = C, embed dim = F
        # batch_first=True => (B, S, E)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=F,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff_mult * F,
            dropout=transformer_dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # fusion
        F_star = max(4, int(round(fusion_ratio * F)))
        self.F_star = F_star
        self.fusion_fc = nn.Linear(n_channels * F, F_star, bias=True)

        # LSTM (1 layer, batch_first)
        self.lstm = nn.LSTM(
            input_size=F_star,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout) if lstm_dropout > 0 else nn.Identity()

        self.attn = TemporalAttention(lstm_hidden)
        self.gamma = nn.Parameter(torch.zeros(()))

        # init (light touch)
        self.apply(lambda m: init_fn(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        if C != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got {C}")

        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        # grouped conv -> (B, C*F, T*)
        x = self.individual_conv(x)
        B, CF, Tstar = x.shape

        # (B, C*F, T*) -> (B, T*, C, F)
        x = x.view(B, self.n_channels, self.F, Tstar).permute(0, 3, 1, 2).contiguous()

        # channel transformer at each time step:
        # (B, T*, C, F) -> (B*T*, C, F) -> enc -> reshape back
        xt = x.view(B * Tstar, self.n_channels, self.F)
        xt = self.channel_transformer(xt)
        x = xt.view(B, Tstar, self.n_channels, self.F)

        # fusion: (B, T*, C*F) -> (B, T*, F*)
        x = x.reshape(B, Tstar, self.n_channels * self.F)
        x = self.fusion_fc(x)

        # LSTM: (B, T*, H)
        h, _ = self.lstm(x)
        h = self.lstm_dropout(h)

        # attention
        c = self.attn(h)          # (B, H)
        last = h[:, -1, :]        # (B, H)
        z = last + self.gamma * c # (B, H)

        return z


class TinyHARClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, head_hidden: int | None = None, init_fn=nn.init.xavier_normal_):
        super().__init__()
        self.backbone = backbone
        out_dim = backbone.lstm.hidden_size  # type: ignore[attr-defined]

        if head_hidden is None:
            self.classifier = nn.Linear(out_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(out_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden, num_classes),
            )

        self.classifier.apply(lambda m: init_fn(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.classifier(z)


class TinyHARModelGenerator(BaseModelGenerator):
    """
    Random TinyHAR-style generator.
    Expects your dataloader to produce x with shape (B, T, C).
    """
    def __init__(self, input_size: int, num_classes: int, init_fn=nn.init.xavier_normal_):
        # input_size == number of channels C, consistent with your RNNModelGenerator usage
        self.input_size = input_size
        self.num_classes = num_classes
        self.init_fn = init_fn

    def sample_arch_config(self) -> dict[str, Any]:
        # Keep ranges close-ish to “TinyHAR-sized” while still exploring.
        F_choices = [4, 6, 8, 12, 16, 24, 32]
        conv_layers_choices = [3, 4, 5]
        kernel_choices = [3, 5, 7]
        stride_choices = [2]  # TinyHAR uses stride 2; keep fixed for comparability

        # transformer: keep tiny; ensure nhead divides F
        transformer_layers_choices = [1, 2]
        transformer_ff_mult_choices = [2, 3, 4]
        transformer_dropout_choices = [0.0, 0.05, 0.1, 0.15]

        # fusion: TinyHAR uses F_star = 2F; allow small variation around it
        fusion_ratio_choices = [1.5, 2.0, 2.5, 3.0]

        # LSTM hidden: often equals F_star in TinyHAR; allow some variation
        lstm_hidden_choices = [16, 24, 32, 48, 64, 96, 128]
        lstm_dropout_choices = [0.0, 0.1, 0.2]

        head_hidden_choices = [None, 128, 256, 512]

        F = random.choice(F_choices)
        nhead = _largest_divisor_leq(F, 4)  # lightweight default

        cfg = {
            "F": F,
            "conv_layers": random.choice(conv_layers_choices),
            "kernel_size": random.choice(kernel_choices),
            "stride": random.choice(stride_choices),
            "transformer_layers": random.choice(transformer_layers_choices),
            "transformer_nhead": nhead,
            "transformer_ff_mult": random.choice(transformer_ff_mult_choices),
            "transformer_dropout": random.choice(transformer_dropout_choices),
            "fusion_ratio": random.choice(fusion_ratio_choices),
            "lstm_hidden": random.choice(lstm_hidden_choices),
            "lstm_dropout": random.choice(lstm_dropout_choices),
            "head_hidden": random.choice(head_hidden_choices),
        }

        return cfg

    def generate_model(self, fixed_arch_config: dict[str, Any] | None = None) -> nn.Module:
        cfg = fixed_arch_config or self.sample_arch_config()

        backbone = TinyHARBackbone(
            n_channels=self.input_size,
            F=int(cfg["F"]),
            conv_layers=int(cfg["conv_layers"]),
            kernel_size=int(cfg["kernel_size"]),
            stride=int(cfg["stride"]),
            transformer_layers=int(cfg["transformer_layers"]),
            transformer_nhead=int(cfg["transformer_nhead"]),
            transformer_ff_mult=int(cfg["transformer_ff_mult"]),
            transformer_dropout=float(cfg["transformer_dropout"]),
            fusion_ratio=float(cfg["fusion_ratio"]),
            lstm_hidden=int(cfg["lstm_hidden"]),
            lstm_dropout=float(cfg["lstm_dropout"]),
            init_fn=self.init_fn,
        )

        model = TinyHARClassifier(
            backbone=backbone,
            num_classes=self.num_classes,
            head_hidden=cfg["head_hidden"],
            init_fn=self.init_fn,
        )
        model._arch_config = cfg
        return model
