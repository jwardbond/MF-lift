# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a paper: a feed-forward neural network (PyTorch Lightning) that predicts microfluidic lift coefficients (`clx`, `cly`) from flow features (`Re`, `ux`, `uy`, `uxx`, `uyy`, `uxy`) for different channel cross-section geometries: rectangular (`r`), triangular (`t`), semicircular (`s`), trapezoidal (`p`).

## Commands

Environment is managed with `uv` (Python >= 3.13; torch is pulled from the CUDA 12.6 index on Windows/Linux).

```sh
uv sync                                   # install deps (add --group dev for seaborn)
uv run python run.py <config.yaml>        # train a single model
uv run python run_multiple.py <folder>    # train every *.yaml in a folder sequentially
uv run python -m src.train --config <config.yaml>   # alternative entry point (note: --config flag, not positional)
uv run tensorboard --logdir outputs/      # view training logs
uv run --group onnx python export_onnx.py checkpoints/rtsp   # export to ONNX (scaler folded in) + verify vs PyTorch
```

There is no test suite, linter config, or build step.

## Architecture

The pipeline is small and config-driven; all four modules in `src/` are wired together by `src/train.py:trainer()`:

- `src/config.py` — Pydantic `Config` loaded from YAML (`run_name`, `paths`, `data`, `model`, `training`). Every YAML key must match these models. See `examples/r_config.yaml` for a full example.
- `src/datamodule.py` — `LiftDataModule` reads CSVs and selects `feature_cols`/`target_cols`. Features are scaled with a `MinMaxScaler` fit on the **training set only** during the `fit` stage; targets are not scaled. `test`/`predict` stages require an already-fitted scaler (pass it in via `scaler=`).
- `src/model.py` — `LiftFNN`: ReLU MLP with sizes from `model.hidden_layers`, MSE loss, Adam + `ReduceLROnPlateau`. Logged metric names are `loss/train` and `loss/val`; checkpointing, early stopping, and the LR scheduler all monitor `loss/val`.
- `src/train.py` — creates `<output_dir>/<run_name>/` with `checkpoints/` and `logs/` (TensorBoard + CSV), trains, then writes `scaler.pkl` / `scaler.yaml` and a dump of the resolved `config.yaml`.

Run output layout (also the format of the committed models in `checkpoints/`):

```
<run_name>/
  checkpoints/   best.ckpt, last.ckpt (training produces e{epoch}-v{loss}.ckpt + last.ckpt)
  logs/
  config.yaml
  scaler.pkl     # pickled sklearn scaler — required to reproduce inputs at inference
  scaler.yaml    # human-readable scaler params
```

Inference pattern (see `results.py`, `examples/lit_comparison.ipynb`): load `config.yaml`, unpickle `scaler.pkl`, `LiftFNN.load_from_checkpoint(...)`, and build a `LiftDataModule` with `scaler=` set and `pred_path` pointing at the CSV to predict on.

## Gotchas

- `trainer()` uses `mkdir(exist_ok=False)`: re-running with an existing `run_name` in the same `output_dir` fails. Change `run_name` or the output dir.
- Paths in config YAMLs resolve relative to the current working directory, not the YAML file. The committed `checkpoints/*/config.yaml` files contain absolute Windows paths from the author's machine.
- `checkpoints/` model names encode the training geometries: `r`, `rt`, `rts`, `rtsp` are trained on combined multi-geometry datasets (`data/combined/<name>_{train,val,test}.csv`); `r0` is trained only on un-augmented rectangular data (used for comparison with Su et al.).
- `data/`, `experiments/`, `outputs/`, `notebooks/`, and `results.py` are gitignored. Data CSVs live in `data/` as `<geom>.csv` plus `<geom>_{train,val,test}.csv` splits; `experiments/<date>/` holds batches of configs for `run_multiple.py`.
