"""Model evaluation component."""

from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
import mlflow
import mlflow.statsmodels
import pandas as pd
from statsmodels.base.model import Results
from statsmodels.regression.quantile_regression import QuantRegResults

from dopro2_HEFTcom_challenge.entity import EvaluationConfig


class Evaluation:
    """Class to evaluate the model."""

    models: list[Results]
    predictions: pd.DataFrame
    sample_data: pd.DataFrame
    score: float

    def __init__(self, config: EvaluationConfig) -> None:
        """
        Constructor for Evaluation class.

        :param config: config values from config.yaml
        """

        self.config = config

    @staticmethod
    def load_models(path: Path) -> list[Results]:
        model_files = Path(path).glob("*.pickle")
        models = []
        for file in model_files:
            models.append(QuantRegResults.load(file))
        return models

    @staticmethod
    def pinball_score(df: pd.DataFrame) -> float:
        def pinball(y, q, alpha):
            return (y - q) * alpha * (y >= q) + (q - y) * (1 - alpha) * (y < q)

        score = []
        for qu in range(10, 100, 10):
            score.append(
                pinball(
                    y=df["total_generation_MWh"],
                    q=df[f"q{qu}"],
                    alpha=qu / 100
                ).mean()
            )
        return sum(score) / len(score)

    def make_predictions(self):
        self.models = self.load_models(self.config.path_to_models)
        test_data = pd.read_parquet(self.config.training_data_path).iloc[400000:]
        self.sample_data = test_data.iloc[:10]
        # nur jetzt zum testen mit iloc
        logger.info("Start making predictions on the trained models.")
        for i, model in enumerate(self.models):
            test_data = test_data.copy()
            test_data[f"q{(i + 1) * 10}"] = model.predict(test_data)
            test_data.loc[test_data[f"q{(i + 1) * 10}"] < 0, f"q{(i + 1) * 10}"] = 0
        self.predictions = test_data[["total_generation_MWh",
                                      "q10", "q20", "q30", "q40", "q50",
                                      "q60", "q70", "q80", "q90"]]
        logger.info("Made predictions on the trained models.")

    def evaluation(self):
        logger.info("Calculate the pinball score on the predictions.")
        self.score = self.pinball_score(self.predictions)
        with open("score.txt", "w", encoding="utf-8") as f:
            f.write(f"Pinball Score: {self.score}")
        logger.info("Score file saved at: score.txt")

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"pinball score": self.score}
            )
            if tracking_url_type_store != "file":
                for i, model in enumerate(self.models):
                    mlflow.statsmodels.log_model(
                        model, "model", registered_model_name=f"q{(i + 1) * 10}",
                        input_example=self.sample_data, signature=False
                    )
            else:
                for _, model in enumerate(self.models):
                    mlflow.statsmodels.log_model(
                        model, "model", input_example=self.sample_data, signature=False
                    )
