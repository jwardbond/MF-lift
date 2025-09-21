import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    train_data: str
    val_data: str
    test_data: str
    pred_data: str
    output_dir: str


class ModelConfig(BaseModel):
    hidden_layers: list[int]


class TrainingConfig(BaseModel):
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float


class Config(BaseModel):
    run_name: str
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            run_name=data["run_name"],
            paths=PathsConfig(**data["paths"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"]),
        )
