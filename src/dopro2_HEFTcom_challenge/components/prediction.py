"""Prediction component."""

from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from dopro2_HEFTcom_challenge.entity import (
    PredictionConfig,
    RebaseAPI
)
from dopro2_HEFTcom_challenge.utils import (
    categorize_wind_dir,
    get_season,
    prep_submission_in_json_format
)


class Prediction:
    """Class to predict on new data from the Rebase API."""

    wind_data: np.ndarray
    solar_data: np.ndarray
    time_df: pd.DataFrame

    def __init__(self, config: PredictionConfig) -> None:
        """
        Constructor for Prediction class.

        :param config: config values from config.yaml
        """

        self.config = config
        self.api = RebaseAPI()

    @staticmethod
    def load_models(
        wind_path: Path,
        solar_path: Path
    ) -> tuple[XGBRegressor, XGBRegressor]:
        """
        Load wind and solar xgboost model.

        :param wind_path: Path to wind model file
        :param solar_path: Path to solar model file
        :return: wind and solar model
        :rtype: tuple[XGBRegressor, XGBRegressor]
        """

        logger.info("Load solar and wind model")

        wind_model_path = list(Path(wind_path)
                               .glob("*.json"))[0]
        wind_model = XGBRegressor()
        wind_model.load_model(wind_model_path)

        solar_model_path = list(Path(solar_path)
                                .glob("*.json"))[0]
        solar_model = XGBRegressor()
        solar_model.load_model(solar_model_path)
        return wind_model, solar_model

    def prepare_data(self) -> None:
        """Get forecast data from rebase api and prepare for prediction"""

        logger.info("Prepare data for the prediction")

        latest_data = self.api.get_latest_forecast_data()

        prediction_data = latest_data.assign(
            season=latest_data["valid_time"].dt.month.apply(get_season),
            wind_dir_cat=latest_data["WindDirection:100"].apply(categorize_wind_dir),
            month=latest_data["valid_time"].dt.month,
            day=latest_data["valid_time"].dt.day,
            hour=latest_data["valid_time"].dt.hour
        )

        season_categories = prediction_data[["season"]]
        season_encoder_path = Path(self.config.components_dir) / "season_encoder"
        season_encoder = joblib.load(season_encoder_path)
        season_encoded = season_encoder.transform(season_categories)
        season_encoded_df = pd.DataFrame(
            season_encoded.toarray(),
            columns=season_encoder.get_feature_names_out()
        )

        windDir_categories = prediction_data[["wind_dir_cat"]]
        windDir_encoder_path = Path(self.config.components_dir) / "windDir_encoder"
        windDir_encoder = joblib.load(windDir_encoder_path)
        windDir_encoded = windDir_encoder.transform(windDir_categories)
        windDir_encoded_df = pd.DataFrame(
            windDir_encoded.toarray(),
            columns=windDir_encoder.get_feature_names_out()
        )

        prediction_data = pd.concat([prediction_data, season_encoded_df,
                                     windDir_encoded_df], axis=1)

        windspeed_pca = prediction_data[["WindSpeed", "WindSpeed:100"]]
        pca_pipe_path = Path(self.config.components_dir) / "scale_pca_pipe_windspeed"
        scale_pca_pipe = joblib.load(pca_pipe_path)
        windspeed_pca = scale_pca_pipe.transform(windspeed_pca)

        prediction_data["WindSpeedPCA"] = windspeed_pca

        wind_features = ["WindSpeedPCA", "hours_after", "season_autumn",
                         "season_spring", "season_summer", "season_winter",
                         "wind_dir_cat_E", "wind_dir_cat_N", "wind_dir_cat_NE",
                         "wind_dir_cat_NW", "wind_dir_cat_S", "wind_dir_cat_SE",
                         "wind_dir_cat_SW", "wind_dir_cat_W"]
        solar_features = ["temp_solar", "CloudCover", "SolarDownwardRadiation",
                          "RelativeHumidity", "hours_after", "month", "day", "hour"]

        self.wind_data = prediction_data[wind_features].to_numpy()
        self.solar_data = prediction_data[solar_features].to_numpy()
        self.time_df = prediction_data[["valid_time"]]

    def predict(self) -> None:
        """Load model and get latest forecast data to make and submit the prediction."""

        wind_model, solar_model = self.load_models(self.config.wind_model_dir,
                                                   self.config.solar_model_dir)

        wind_predictions = wind_model.predict(self.wind_data)
        wind_predictions[wind_predictions < 0] = 0
        solar_predictions = solar_model.predict(self.solar_data)
        solar_predictions[solar_predictions < 0] = 0
        predictions_all = wind_predictions + solar_predictions
        predictions_all.sort(axis=1)

        quantil_cols = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        prediction_df = pd.DataFrame(predictions_all, columns=quantil_cols)

        submission_data = self.time_df.join(prediction_df)
        submission_data["market_bid"] = submission_data["q50"]

        submission_data_json = prep_submission_in_json_format(submission_data)
        print(submission_data_json)

        # self.api.submit(submission_data_json)
