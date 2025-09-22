import lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LiftFNN(L.LightningModule):
    def __init__(
        self,
        hidden_layers: list[int],
        learning_rate: float,
        lr_scheduler_patience: int,
        lr_scheduler_factor: float,
    ) -> None:
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(8, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], 3))

        self.stack = nn.Sequential(*layers)

        # Training hyperparams
        self.learning_rate = learning_rate
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

    def forward(self, x):
        return self.stack(x)

    def training_step(self, batch, batch_idx):
        stage = "train"
        loss = self._common_step(batch, batch_idx)
        self._common_log(stage, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        stage = "val"
        loss = self._common_step(batch, batch_idx)
        self._common_log(stage, loss)

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=1e-6,
            ),
            "monitor": "loss/val",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])

    def _common_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss

    def _common_log(self, stage: str, loss):
        assert stage in ("train", "val", "test")
        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True)
