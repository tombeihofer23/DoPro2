"""Main program."""

from typing import Final

from loguru import logger

from dopro2_HEFTcom_challenge.config import config_logger
from dopro2_HEFTcom_challenge.pipeline import DataIngestionTrainingPipeline


STAGE_NAME: Final = "Data Ingestion stage"

# set up logging
config_logger()

try:
    logger.info(">>> stage {} started <<<", STAGE_NAME)
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(">>> stage {} completed <<<", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e
