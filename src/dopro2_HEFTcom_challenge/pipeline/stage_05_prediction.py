"""Fifth ML Pipeline stage: predict on new data."""

from typing import Final

from loguru import logger

from dopro2_HEFTcom_challenge.components import Prediction
from dopro2_HEFTcom_challenge.config import ConfigurationManager


STAGE_NAME: Final = "Prediction stage"


class PredictionPipeline:
    """Pipeline that make predictions on you data from Rebase API."""

    def __init__(self):
        pass

    def main(self) -> None:
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()
        prediction = Prediction(config=prediction_config)
        prediction.prepare_data()
        prediction.predict()


if __name__ == "__main__":
    try:
        logger.info(">>> stage {} started <<<", STAGE_NAME)
        obj = PredictionPipeline()
        obj.main()
        logger.info(">>> stage {} completed <<<", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
