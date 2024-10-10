"""Manage all configurations."""

import os
from pathlib import Path

from loguru import logger
import yaml

from dopro2_HEFTcom_challenge.constants import (
    PARAMS_FILE_PATH,
    CONFIG_FILE_PATH
)
from dopro2_HEFTcom_challenge.entity import (
    DataIngestionConfig,
    DataPreparationConfig
)


class ConfigurationManager:
    """Class to manage all configurations."""

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ) -> None:
        """
        Constructor for ConfigurationManager Class.
        Creates artifacts folder.

        :param config_filepath: Path to config.yaml file
        :param params_filepath: Path to params.yaml file

        """
        with config_filepath.open("r", encoding="utf-8") as f:
            self.config: dict = yaml.safe_load(f)

        with params_filepath.open("r", encoding="utf-8") as f:
            self.params: dict = yaml.safe_load(f)

        os.makedirs(self.config["artifacts_root"], exist_ok=True)
        logger.info("created directory at: {}", self.config["artifacts_root"])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get all config params and create folder in artifacts dir.

        :return: values from config.yaml
        :rtype: DataIngestionConfig
        """
        config = self.config["data_ingestion"]

        os.makedirs(config["root_dir"], exist_ok=True)
        logger.info("created directory at: {}", config["root_dir"])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_url=config["source_url"]
        )

        return data_ingestion_config

    def get_data_preparation_config(self) -> DataPreparationConfig:
        """
        Get all config params and create folder in artifacts dir.

        :return: values from config.yaml
        :rtype: DataPreparationConfig
        """
        config = self.config["data_preparation"]

        os.makedirs(config["root_dir"], exist_ok=True)
        logger.info("created directory at: {}", config["root_dir"])

        data_preparation_config = DataPreparationConfig(
            root_dir=config["root_dir"],
            weather_data_path=config["weather_data_path"],
            energy_data_path=config["energy_data_path"],
            training_data_path=config["training_data_path"],
            test_data_path=config["test_data_path"]
        )

        return data_preparation_config
