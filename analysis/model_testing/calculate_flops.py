import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from fvcore.nn import FlopCountAnalysis
from tqdm.auto import tqdm

from utils.path_utils import setup_path
setup_path()

from analysis.post_processing_pipeline.zero_cost_scores_parallel import load_checkpoint, set_seed
from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS
from data.dataset_registry import dataset_registry

DATASETS = ["motionsense", "hhar", "rwhar", "pamap2", "mobiactv2", "myogym"]


def _parse_dataset_key(dataset_name: str) -> tuple[str, float]:
    if "_" in dataset_name:
        base, rho_str = dataset_name.split("_", 1)
        return base, float(rho_str)
    return dataset_name, 1.0


def count_cnn_flops(model: torch.nn.Module, input_shape: tuple[int, ...]) -> float:
    model = model.eval().cpu()
    x = torch.randn(1, *input_shape)

    total_flops = 0

    def conv1d_hook(module, inputs, output):
        nonlocal total_flops

        # input:  [batch, in_channels, seq_len]
        # output: [batch, out_channels, out_seq_len]
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_seq_len = output.shape[2]

        kernel_size = module.kernel_size[0]
        in_channels = module.in_channels
        groups = module.groups

        # multiply-add convention: 2 FLOPs
        flops = (
            batch_size
            * out_channels
            * out_seq_len
            * kernel_size
            * (in_channels // groups)
            * 2
        )

        if module.bias is not None:
            flops += batch_size * out_channels * out_seq_len

        total_flops += flops

    def linear_hook(module, inputs, output):
        nonlocal total_flops

        batch_size = output.shape[0] if output.ndim > 1 else 1
        flops = batch_size * module.in_features * module.out_features * 2

        if module.bias is not None:
            flops += batch_size * module.out_features

        total_flops += flops

    handles = []

    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            handles.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        _ = model(x)

    for handle in handles:
        handle.remove()

    return float(total_flops)


def count_lstm_flops(model: torch.nn.Module, seq_len: int, input_size: int) -> float:
    """
    Approx forward FLOPs for nn.LSTM modules.

    Per layer, per timestep:
        4 gates * (input matmul + hidden matmul + bias-ish ops)
        approx = 4 * hidden_size * (input_dim + hidden_size) * 2

    Factor 2 counts multiply + add.
    """
    total = 0.0

    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            hidden_size = module.hidden_size
            num_layers = module.num_layers
            bidirectional = module.bidirectional
            directions = 2 if bidirectional else 1

            layer_input_size = module.input_size

            for layer_idx in range(num_layers):
                for _ in range(directions):
                    total += seq_len * 4 * hidden_size * (layer_input_size + hidden_size) * 2

                # after first layer, input to next layer is hidden_size * directions
                layer_input_size = hidden_size * directions

    return float(total)


def count_transformer_flops(model: torch.nn.Module, seq_len: int) -> float:
    """
    Approx forward FLOPs for nn.TransformerEncoderLayer.

    For each encoder layer:
      QKV projections:      3 * 2 * T * d_model * d_model
      attention scores:     2 * T * T * d_model
      attention values:     2 * T * T * d_model
      output projection:    2 * T * d_model * d_model
      FFN linear1:          2 * T * d_model * dim_ff
      FFN linear2:          2 * T * dim_ff * d_model

    Ignores small elementwise/dropout/layernorm costs.
    """
    total = 0.0

    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            d_model = module.self_attn.embed_dim
            dim_ff = module.linear1.out_features
            T = seq_len

            qkv = 3 * 2 * T * d_model * d_model
            attn_scores = 2 * T * T * d_model
            attn_weighted_values = 2 * T * T * d_model
            out_proj = 2 * T * d_model * d_model
            ffn = 2 * T * d_model * dim_ff + 2 * T * dim_ff * d_model

            total += qkv + attn_scores + attn_weighted_values + out_proj + ffn

    return float(total)


def count_model_flops(model: torch.nn.Module, model_type: str, input_shape: tuple[int, ...]) -> float:
    seq_len, input_size = input_shape

    model_type_lower = model_type.lower()

    if "cnn" in model_type_lower:
        return count_cnn_flops(model, input_shape)

    if "lstm" in model_type_lower or "rnn" in model_type_lower:
        return count_lstm_flops(model, seq_len=seq_len, input_size=input_size)

    if "transformer" in model_type_lower:
        return count_transformer_flops(model, seq_len=seq_len)

    # fallback for any weird model type
    return count_cnn_flops(model, input_shape)


def _worker(row_dict: dict, *, num_classes: int, input_size: int, input_shape: tuple[int, ...]) -> dict:
    set_seed(int(row_dict["arch_seed"]))

    model, _ = load_checkpoint(
        Path(row_dict["model_path"]),
        num_classes=num_classes,
        input_size=input_size,
        arch=row_dict["model_type"],
    )

    flops = count_model_flops(
        model=model,
        model_type=row_dict["model_type"],
        input_shape=input_shape,
    )

    return {
        "run_dir": row_dict["run_dir"],
        "model_path": str(row_dict["model_path"]),
        "model_type": row_dict["model_type"],
        "arch_seed": row_dict["arch_seed"],
        "flops": flops,
        "gflops": flops / 1e9,
    }


def main(max_workers: int | None = None):
    set_seed(42)

    if max_workers is None:
        max_workers = max(1, min(16, (os.cpu_count() or 1) - 10))

    for dataset_name in DATASETS:
        print(f"\n=== {dataset_name} ===")
        cfg = DATASET_CONFIGS[dataset_name]
        zc_path = Path(cfg["zero_cost_pkl"])

        if not zc_path.exists():
            print(f"  SKIP: {zc_path} not found")
            continue

        zc_df = pd.read_pickle(zc_path)
        print(f"  {len(zc_df)} models")

        out = Path(cfg["model_flops_pkl"])
        out.parent.mkdir(parents=True, exist_ok=True)

        base_name, rho = _parse_dataset_key(dataset_name)
        seed = cfg.get("seed", 42)

        ds = dataset_registry.get_dataset(base_name)(
            phase="train",
            root_dir=cfg["data_root"],
            rho=rho,
            seed=seed,
        )

        num_classes = getattr(ds, "num_classes", len(set(ds.labels)))
        input_size = ds.config["input_size"]

        # IMPORTANT: model expects (batch, seq_len, input_size)
        input_shape = (1000, input_size)

        rows = zc_df[["run_dir", "model_path", "model_type", "arch_seed"]].to_dict("records")

        records: list[dict] = []

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _worker,
                    r,
                    num_classes=num_classes,
                    input_size=input_size,
                    input_shape=input_shape,
                )
                for r in rows
            ]

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  {dataset_name}"):
                records.append(fut.result())

        flops_df = pd.DataFrame(records)
        flops_df.to_pickle(out)

        print(f"  Saved {len(flops_df)} rows -> {out}")
        print(flops_df.groupby("model_type")["gflops"].describe())


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()