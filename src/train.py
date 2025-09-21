import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml

from src.config import Config
from src.datamodule import LiftDataModule
from src.model import LiftFNN


def trainer(config: Config):
    # Create output directories
    output_dir = Path(config.paths.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)

    # Init data and model
    data_module = LiftDataModule(
        train_path=config.paths.train_data,
        val_path=config.paths.val_data,
        test_path=config.paths.test_data,
        pred_path=config.paths.pred_data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    model = LiftFNN(hidden_layers=config.model.hidden_layers)

    # Init logging
    tb_logger = TensorBoardLogger(save_dir=logs_dir, name="", version="")
    csv_logger = CSVLogger(save_dir=logs_dir, name="", version="")

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_top_k=2,
        verbose=True,
        monitor="loss/val",
        mode="min",
        save_last=True,
    )

    # Train
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        default_root_dir=output_dir,
        callbacks=[checkpoint_cb],
        logger=[tb_logger, csv_logger],
        # accelerator="cpu",
    )
    trainer.fit(model, datamodule=data_module)

    # Save outputs
    data_module.save_scaler(output_dir)
    with output_dir.joinpath("config.yaml").open("w") as f:
        yaml.dump(config, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config.from_yaml(args.config)

    trainer(config)
