import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    train_data: str
    val_data: str
    test_data: str
    pred_data: str
    output_dir: str


class DataConfig(BaseModel):
    batch_size: int
    num_workers: int
    feature_cols: list[str]
    target_cols: list[str]


class ModelConfig(BaseModel):
    hidden_layers: list[int]


class TrainingConfig(BaseModel):
    max_epochs: int
    learning_rate: float
    lr_scheduler_patience: int
    lr_scheduler_factor: float
    early_stop_patience: int


class Config(BaseModel):
    run_name: str
    paths: PathsConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
