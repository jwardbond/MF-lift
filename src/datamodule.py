from pathlib import Path
import pickle

import lightning as L
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class LiftDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        pred_path,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.pred_path = pred_path

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.feature_cols = [
            "Re",
            "G1",
            "ux_G1",
            "uy_G1",
            "G2",
            "uxx_G2",
            "uyy_G2",
            "uxy_G2",
        ]
        self.target_cols = ["logCL", "nx", "ny"]

        self.scaler = StandardScaler()

    def setup(self, stage: str):
        if stage == "fit":
            train_df = pd.read_csv(self.train_path)
            self.scaler.fit(train_df[self.feature_cols].to_numpy())

            val_df = pd.read_csv(self.val_path)

            self.train_set = self._create_dataset(train_df)
            self.val_set = self._create_dataset(val_df)

        elif stage == "test":
            test_df = pd.read_csv(self.test_path)
            self.test_set = self._create_dataset(val_df)

        elif stage == "pred":
            pred_df = pd.read_csv(self.pred_path)
            self.pred_set = self._create_dataset(pred_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def save_scaler(self, dir: Path):
        """Save the scaler params and scaler sklearn object."""

        means = self.scaler.mean_.tolist()
        stdev = self.scaler.scale_.tolist()
        variance = self.scaler.var_.tolist()

        scaler_dict = {
            "mean": means,
            "stdev": stdev,
            "variance": variance,
            "feature_cols": self.feature_cols,
        }

        yaml_path = dir / "scaler.yaml"
        with yaml_path.open("w") as f:
            yaml.safe_dump(scaler_dict, f)

        pickle_path = dir / "scaler.pkl"
        with pickle_path.open("wb") as f:
            pickle.dump(self.scaler, f)

    def _create_dataset(self, df):
        """Create a TensorDataset from a DataFrame."""

        X = df[self.feature_cols].to_numpy()
        X_scaled = self.scaler.transform(X)

        y = df[self.target_cols].to_numpy()

        return TensorDataset(
            torch.from_numpy(X_scaled).float(), torch.from_numpy(y).float()
        )
