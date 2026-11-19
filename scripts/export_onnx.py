"""Export a trained LiftFNN run to ONNX, with the input MinMaxScaler folded into the graph.

The exported model takes raw (unscaled) features in the order given by the run's
`feature_cols` and returns the raw targets (`target_cols`). The column orders are also
stored in the ONNX metadata (`feature_cols`, `target_cols`).

Usage:
    uv sync --group onnx
    uv run python export_onnx.py checkpoints/rtsp                 # -> checkpoints/rtsp/rtsp.onnx
    uv run python export_onnx.py checkpoints/rtsp -o lift.onnx --ckpt last.ckpt
    uv run python export_onnx.py checkpoints/rtsp --csv data/combined/rtsp_test.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from src.config import Config
from src.model import LiftFNN


class ScaledLiftFNN(nn.Module):
    """LiftFNN with MinMaxScaler applied first: forward(x) = stack(x * scale + min)."""

    def __init__(self, model: LiftFNN, scaler: MinMaxScaler) -> None:
        super().__init__()
        self.stack = model.stack
        self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        self.register_buffer("offset", torch.tensor(scaler.min_, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x * self.scale + self.offset)


def load_run(run_dir: Path, ckpt_name: str) -> tuple[Config, MinMaxScaler, LiftFNN]:
    config = Config.from_yaml(run_dir / "config.yaml")
    with (run_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    if not isinstance(scaler, MinMaxScaler):
        raise TypeError(f"Expected a MinMaxScaler, got {type(scaler).__name__}")

    model = LiftFNN.load_from_checkpoint(
        run_dir / "checkpoints" / ckpt_name, map_location="cpu"
    )
    model.eval()
    return config, scaler, model


def export(module: nn.Module, n_inputs: int, out_path: Path, config: Config) -> None:
    dummy = torch.zeros(2, n_inputs, dtype=torch.float32)
    batch = torch.export.Dim("batch")
    program = torch.onnx.export(
        module,
        (dummy,),
        input_names=["features"],
        output_names=["lift"],
        dynamic_shapes={"x": {0: batch}},
        dynamo=True,
        verbose=False,  # its progress output contains emoji that crash cp1252 consoles
    )
    program.optimize()
    program.save(str(out_path))

    # Record column order so consumers don't need config.yaml
    model_proto = onnx.load(str(out_path))
    meta = {
        "feature_cols": ",".join(config.data.feature_cols),
        "target_cols": ",".join(config.data.target_cols),
        "run_name": config.run_name,
    }
    for key, value in meta.items():
        entry = model_proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(out_path))


def verify(
    out_path: Path,
    model: LiftFNN,
    scaler: MinMaxScaler,
    config: Config,
    csv_path: Path | None,
    n_samples: int = 10_000,
) -> None:
    if csv_path is not None:
        x = pd.read_csv(csv_path)[config.data.feature_cols].to_numpy(dtype=np.float32)
        source = str(csv_path)
    else:
        rng = np.random.default_rng(0)
        x = rng.uniform(
            scaler.data_min_, scaler.data_max_, size=(n_samples, scaler.n_features_in_)
        ).astype(np.float32)
        source = "uniform samples over the scaler's training range"

    # Reference: the existing pipeline (sklearn scaler -> PyTorch model)
    x_scaled = scaler.transform(x)  # scaler was fitted on a bare array
    with torch.no_grad():
        expected = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"features": x})[0]

    # Differences are float32 rounding (scaling done in float32 vs sklearn's float64), so
    # compare absolute error against the output magnitude rather than per-element rel. error.
    max_err = np.abs(actual - expected).max()
    max_out = np.abs(expected).max()
    print(f"Verified {len(x)} rows from {source}")
    print(f"  max abs error {max_err:.3e} (max |output| {max_out:.3e})")
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5 * max_out)
    print("  OK: ONNX output matches PyTorch")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="Run directory, e.g. checkpoints/rtsp")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: <run_dir>/<run_name>.onnx)")
    parser.add_argument("--ckpt", default="best.ckpt", help="Checkpoint file in <run_dir>/checkpoints/")
    parser.add_argument("--csv", type=Path, help="CSV to verify against (default: random samples)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, scaler, model = load_run(args.run_dir, args.ckpt)
    out_path = args.output or args.run_dir / f"{config.run_name}.onnx"

    export(ScaledLiftFNN(model, scaler).eval(), len(config.data.feature_cols), out_path, config)
    print(f"Exported {args.run_dir / 'checkpoints' / args.ckpt} -> {out_path}")

    verify(out_path, model, scaler, config, args.csv)
