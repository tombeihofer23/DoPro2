"""Main program."""

from typing import Final

from loguru import logger

from dopro2_HEFTcom_challenge.config import config_logger
from dopro2_HEFTcom_challenge.pipeline import (
    # DataIngestionTrainingPipeline,
    # DataPreparationTrainingPipeline,
    # ModelEvaluationPipeline,
    # ModelTrainingPipeline,
    PredictionPipeline
)


# STAGE_NAME_01: Final = "Data Ingestion stage"

# set up logging
config_logger()

# try:
#     logger.info(">>> stage {} started <<<", STAGE_NAME_01)
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(">>> stage {} completed <<<", STAGE_NAME_01)
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME_02: Final = "Data Preparation stage"

# try:
#     logger.info(">>> stage {} started <<<", STAGE_NAME_02)
#     obj = DataPreparationTrainingPipeline()
#     obj.main()
#     logger.info(">>> stage {} completed <<<", STAGE_NAME_02)
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME_03: Final = "Model training stage"

# try:
#     logger.info(">>> stage {} started <<<", STAGE_NAME_03)
#     obj = ModelTrainingPipeline()
#     obj.main()
#     logger.info(">>> stage {} completed <<<", STAGE_NAME_03)
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME_04: Final = "Model evaluation stage"

# try:
#     logger.info(">>> stage {} started <<<", STAGE_NAME_04)
#     obj = ModelEvaluationPipeline()
#     obj.main()
#     logger.info(">>> stage {} completed <<<", STAGE_NAME_04)
# except Exception as e:
#     logger.exception(e)
#     raise e


STAGE_NAME_05: Final = "Prediction stage"

try:
    logger.info(">>> stage {} started <<<", STAGE_NAME_05)
    obj = PredictionPipeline()
    obj.main()
    logger.info(">>> stage {} completed <<<", STAGE_NAME_05)
except Exception as e:
    logger.exception(e)
    raise e
