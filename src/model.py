import lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


class LiftFNN(L.LightningModule):
    def __init__(self, hidden_layers: list[int]) -> None:
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
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = {
            "scheduler": StepLR(optimizer, step_size=200, gamma=0.1),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _common_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss

    def _common_log(self, stage: str, loss):
        assert stage in ("train", "val", "test")
        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True)
