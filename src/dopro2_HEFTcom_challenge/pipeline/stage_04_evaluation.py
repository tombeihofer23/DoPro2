"""Fourth ML Pipeline stage: model evaluation."""

from typing import Final

from loguru import logger

from dopro2_HEFTcom_challenge.components import Evaluation
from dopro2_HEFTcom_challenge.config import ConfigurationManager

import dagshub
dagshub.init(  # type: ignore
    repo_owner="tombeihofer23", repo_name="DoPro2", mlflow=True
)


STAGE_NAME: Final = "Model evaluation stage"


class ModelEvaluationPipeline:
    """Pipeline that evaluate the model."""

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.make_predictions()
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(">>> stage {} started <<<", STAGE_NAME)
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(">>> stage {} completed <<<", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
