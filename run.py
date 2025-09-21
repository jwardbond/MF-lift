import argparse

from src.config import Config
from src.train import trainer

import torch

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config.from_yaml(args.config)

    trainer(config)
