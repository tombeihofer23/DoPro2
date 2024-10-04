"""Create directory structure."""

import os
from pathlib import Path
from loguru import logger


PROJECT_NAME = "dopro2_HEFTcom_challenge"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{PROJECT_NAME}/__init__.py",
    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/configuration.py",
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/utils/__init__.py",
    "config/config.yaml"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logger.info("Creating directory: {} for the file: {}",
                    filedir, filename)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w", encoding="utf-8") as f:
            logger.info("Creating emtpy file: {}", filepath)

    else:
        logger.info("{} already exists", filename)
