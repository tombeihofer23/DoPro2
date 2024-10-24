"""Class for config params."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """Entity-Class for config params."""

    root_dir: Path
    """Directory into which data will be loaded."""

    source_url: str
    """URL where the data is located."""


@dataclass(frozen=True)
class DataPreparationConfig:
    """Entity-Class for data preparation config params."""

    root_dir: Path
    """Directory into which data will be loaded."""

    weather_data_path: Path
    """Directory where raw weather data is stored."""

    energy_data_path: Path
    """Directory where raw energy data is stored."""

    training_data_path: Path
    """Directory into which training data will be loaded."""

    test_data_path: Path
    """Directory into which test data will be loaded."""

    components_path: Path
    """Directory where preprocessing components are stored."""


@dataclass(frozen=True)
class TrainingConfig:
    """Entity-Class for training config params."""

    root_dir: Path
    """Directory into which data will be loaded."""

    trained_models_path: Path
    """Directory where trained models will be stored."""

    training_data_path: Path
    """Directory where training data is stored."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Entity-Class for data evaluation config params."""

    path_to_models: Path
    """Directory in which models are stored."""

    training_data_path: Path
    """Directory where training data is stored."""

    all_params: dict
    """Model parameters."""

    mlflow_uri: str
    """URL to MLFlow dashboard."""
