"""Third ML Pipeline stage: model training."""

from typing import Final

from loguru import logger

from dopro2_HEFTcom_challenge.components import Training
from dopro2_HEFTcom_challenge.config import ConfigurationManager


STAGE_NAME: Final = "Model training stage"


class ModelTrainingPipeline:
    """Pipeline that train the model."""

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.train()


if __name__ == "__main__":
    try:
        logger.info(">>> stage {} started <<<", STAGE_NAME)
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(">>> stage {} completed <<<", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
