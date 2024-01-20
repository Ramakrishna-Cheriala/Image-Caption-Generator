from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_api: Path
    local_files_dir: Path
    unzip_dir: Path
    batch_size: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    model_dir: Path
    pkl_file: Path
    file_path: Path
    saved_model_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    train_dir: Path
    pkl_file: Path
    saved_trained_model_dir: Path
    epochs: int
    batch_size: int
