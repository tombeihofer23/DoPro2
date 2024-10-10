"""Model trainig component."""

import os
from pathlib import Path

from loguru import logger
import pandas as pd
import statsmodels.formula.api as smf

from dopro2_HEFTcom_challenge.entity import TrainingConfig


class Training:
    """Class to performe the model training."""

    def __init__(self, config: TrainingConfig) -> None:
        """
        Constructor for Training class.

        :param config: config values from config.yaml
        """

        self.config = config

    @staticmethod
    def save_models(forecast_models: dict, path: Path) -> None:
        os.makedirs(path, exist_ok=True)
        logger.info("created directory at: {}", path)

        for quantile in range(10, 100, 10):
            forecast_models[f"q{quantile}"].save(f"{path}/model_q{quantile}.pickle")
        logger.info("saved all models in at {}", path)

    def train(self) -> None:
        logger.info("Loading trainind data from {}", self.config.training_data_path)
        training_data = pd.read_parquet(self.config.training_data_path)
        model = smf.quantreg(
            formula="total_generation_MWh ~ bs(SolarDownwardRadiation,df=5) \
                + bs(WindSpeed,df=8)",
            data=training_data
        )

        logger.info("Start model training")
        forecast_models = {}
        for quantile in range(10, 100, 10):
            forecast_models[f"q{quantile}"] = model.fit(q=quantile / 100, max_iter=2500)
            training_data[f"q{quantile}"] = forecast_models[f"q{quantile}"]\
                .predict(training_data)
            training_data.loc[training_data[f"q{quantile}"] < 0, f"q{quantile}"] = 0
        logger.info("Model trained")

        self.save_models(forecast_models, self.config.trained_models_path)
