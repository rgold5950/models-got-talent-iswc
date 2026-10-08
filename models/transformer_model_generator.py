# models/transformer_model_generator.py
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from models.base_model_generator import BaseModelGenerator


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Classic sinusoidal positional embedding.

    Returns (1, T, D) so it can be broadcast across batch.
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t <= self.pe.shape[1]:
            return self.pe[:, :t, :].to(device=device, dtype=dtype)

        # If sequence is longer than max_len, rebuild just-in-time
        pe = self._build_pe(t, self.d_model).to(device=device, dtype=dtype)
        return pe


class VanillaTransformerBackbone(nn.Module):
    """
    (B, T, C) -> proj -> +sinusoidal pos -> TransformerEncoder -> pool -> (B, D)
    """
    def __init__(
        self,
        n_channels: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        ff_mult: int,
        dropout: float,
        attn_dropout: float,
        activation: str,
        norm_first: bool,
        pooling: str,              # "mean" | "last" | "cls"
        use_input_layernorm: bool,
        max_len: int,
        init_fn=nn.init.xavier_uniform_,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.pooling = pooling

        self.in_proj = nn.Linear(n_channels, d_model, bias=True)
        self.in_ln = nn.LayerNorm(d_model) if use_input_layernorm else nn.Identity()
        self.pos = SinusoidalPositionalEmbedding(d_model=d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=min(int(ff_mult * d_model), 512),  # Cap FFN inner dimension at 512
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        # PyTorch uses `dropout` inside attention too; we allow separate knob by patching if desired.
        # Keep it simple: set attention dropout via internal module if present.
        if hasattr(enc_layer, "self_attn") and hasattr(enc_layer.self_attn, "dropout"):
            try:
                enc_layer.self_attn.dropout = float(attn_dropout)  # type: ignore[attr-defined]
            except Exception:
                pass

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # init
        init_fn(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, T, C); got {tuple(x.shape)}")
        b, t, c = x.shape
        if c != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got C={c}")

        h = self.in_proj(x)  # (B, T, D)
        h = self.in_ln(h)

        if self.cls_token is not None:
            cls = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
            h = torch.cat([cls, h], dim=1)          # (B, 1+T, D)
            pos = self.pos(t + 1, device=h.device, dtype=h.dtype)
        else:
            pos = self.pos(t, device=h.device, dtype=h.dtype)

        h = self.drop(h + pos)
        h = self.encoder(h)  # (B, S, D)

        if self.pooling == "mean":
            if self.cls_token is not None:
                h = h[:, 1:, :]  # drop CLS for mean over real timesteps
            return h.mean(dim=1)
        if self.pooling == "last":
            return h[:, -1, :]
        if self.pooling == "cls":
            return h[:, 0, :]

        raise ValueError(f"Unknown pooling={self.pooling!r}")


class VanillaTransformerClassifier(nn.Module):
    def __init__(self, backbone: VanillaTransformerBackbone, num_classes: int, init_fn=nn.init.xavier_uniform_):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, num_classes)
        init_fn(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)     # (B, D)
        logits = self.head(z)    # (B, K)
        return logits            # training should apply CE/softmax externally


class TransformerModelGenerator(BaseModelGenerator):
    """
    Random vanilla Transformer generator.
    Expects dataloader to produce x with shape (B, T, C).
    """
    def __init__(self, input_size: int, num_classes: int, init_fn=nn.init.xavier_uniform_):
        self.input_size = input_size
        self.num_classes = num_classes
        self.init_fn = init_fn

    def sample_arch_config(self) -> dict:
        # Key knobs that matter most across HAR datasets:
        # - d_model (capacity)
        # - num_layers (depth)
        # - nhead (attention granularity)
        # - ff_mult (MLP capacity)
        # - dropout/attn_dropout (regularization)
        # - pooling ("mean" often strong; "cls" sometimes helps)
        d_model = random.choice([64, 128, 256])

        num_layers = random.choice([2, 3, 4])

        # heads must divide d_model. In practice 2/4/8 are the sweet spot.
        head_candidates = [h for h in [2, 4, 8] if d_model % h == 0]
        if not head_candidates:
            head_candidates = _divisors(d_model)
        nhead = random.choice(head_candidates)

        ff_mult = random.choice([2, 4, 8])
        dropout = random.choice([0.0, 0.05, 0.1, 0.15, 0.2])
        attn_dropout = random.choice([0.0, 0.05, 0.1])

        activation = random.choice(["relu", "gelu"])
        norm_first = True  # tends to be stable for deeper encoders

        pooling = random.choice(["mean", "last", "cls"])
        use_input_layernorm = random.choice([True, False])

        # Sequence lengths vary by dataset/windowing; keep generous.
        max_len = random.choice([512, 1024, 2048, 4096])

        cfg = {
            "d_model": d_model,
            "num_layers": num_layers,
            "nhead": nhead,
            "ff_mult": ff_mult,
            "dropout": float(dropout),
            "attn_dropout": float(attn_dropout),
            "activation": activation,
            "norm_first": bool(norm_first),
            "pooling": pooling,
            "use_input_layernorm": bool(use_input_layernorm),
            "max_len": int(max_len),
        }
        return cfg

    def generate_model(self, fixed_arch_config: dict | None = None) -> nn.Module:
        cfg = fixed_arch_config or self.sample_arch_config()

        backbone = VanillaTransformerBackbone(
            n_channels=self.input_size,
            d_model=int(cfg["d_model"]),
            num_layers=int(cfg["num_layers"]),
            nhead=int(cfg["nhead"]),
            ff_mult=int(cfg["ff_mult"]),
            dropout=float(cfg["dropout"]),
            attn_dropout=float(cfg["attn_dropout"]),
            activation=str(cfg["activation"]),
            norm_first=bool(cfg["norm_first"]),
            pooling=str(cfg["pooling"]),
            use_input_layernorm=bool(cfg["use_input_layernorm"]),
            max_len=int(cfg["max_len"]),
            init_fn=self.init_fn,
        )

        model = VanillaTransformerClassifier(backbone=backbone, num_classes=self.num_classes, init_fn=self.init_fn)
        model._arch_config = cfg  # match your pattern
        return model
