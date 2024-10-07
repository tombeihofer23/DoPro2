"""Data ingestion component."""

import gdown
from loguru import logger

from dopro2_HEFTcom_challenge.entity import DataIngestionConfig


class DataIngestion:
    """Class to performe data ingestion."""

    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Constructor for DataIngestion class.

        :param config: config values from config.yaml
        """

        self.config = config

    def download_files(self) -> None:
        """Fetch data from source url"""

        try:
            data_url: str = self.config.source_url
            download_dir: str = self.config.root_dir
            logger.info("Downloading data from {} into folder {}",
                        data_url, download_dir)

            gdown.download_folder(data_url, output=download_dir)
            logger.info("Downloaded data from {} into folder {}",
                        data_url, download_dir)
        except Exception as e:
            raise e
