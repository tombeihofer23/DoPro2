"""Configure logging for project."""

from pathlib import Path
from typing import Final

from loguru import logger

__all__: list[str] = ["config_logger"]

LOG_FILE: Final = Path("log") / "running_logs.log"


def config_logger() -> None:
    """Configure logging."""
    logger.add(LOG_FILE, rotation="1 MB")
