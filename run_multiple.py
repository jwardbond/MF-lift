import argparse
from pathlib import Path

from src.config import Config
from src.train import trainer

import torch

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing config files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_dir = Path(args.folder)

    for file in config_dir.glob("*.yaml"):
        config = Config.from_yaml(file)
        trainer(config)
