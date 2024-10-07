"""First ML Pipeline stage: data ingestion."""

from typing import Final

from loguru import logger

from dopro2_HEFTcom_challenge.components import DataIngestion
from dopro2_HEFTcom_challenge.config import ConfigurationManager


STAGE_NAME: Final = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_files()


if __name__ == "__main__":
    try:
        logger.info(">>> stage {} started <<<", STAGE_NAME)
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(">>> stage {} completed <<<", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
