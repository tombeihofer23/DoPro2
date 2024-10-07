"""Store project constansts."""

from pathlib import Path
from typing import Final

_all__: list[str] = [
    "CONFIG_FILE_PATH",
    "PARAMS_FILE_PATH"
]

CONFIG_FILE_PATH: Final = Path("config/config.yaml")
PARAMS_FILE_PATH: Final = Path("params.yaml")
