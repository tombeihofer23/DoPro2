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
