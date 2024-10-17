"""Data preparation component."""

import os
from pathlib import Path
from typing import Final

from loguru import logger
# import numpy as np
import pandas as pd
import re
# import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from dopro2_HEFTcom_challenge.entity import DataPreparationConfig
from dopro2_HEFTcom_challenge.utils import load_weather_data


class DataPreparation:
    """Class to performe data preparation."""

    def __init__(self, config: DataPreparationConfig) -> None:
        """
        Constructor for DataPreparation class.

        :param config: config values from config.yaml
        """

        self.config = config

    def cleaning_energy_data(self) -> pd.DataFrame:

        logger.info("Start cleaning energy data")
        energy_files = Path(self.config.energy_data_path).glob("*.csv")
        df_raw = pd.concat(
            (pd.read_csv(f) for _, f in enumerate(energy_files)),
            ignore_index=True
        )
        df = (
            df_raw
            .assign(dtm=pd.to_datetime(df_raw["dtm"]),
                    Wind_MWh_credit=0.5 * df_raw["Wind_MW"] - df_raw["boa_MWh"],
                    Solar_MWh_credit=0.5 * df_raw["Solar_MW"]
                    )
        )

        df.to_parquet(f"{self.config.root_dir}/energy_processed.parquet")
        logger.info("Cleaned energy data: file safed under {}", self.config.root_dir)

        return df

    def cleaning_weather_data(self):
        logger.info("Start cleaning weather data")
        weather_files = sorted(Path(self.config.weather_data_path).glob("*.nc"))

        dwd_hornsea_rx = re.compile("dwd_icon_eu_hornsea")
        dwd_hornsea_files = [f for f in weather_files if dwd_hornsea_rx.match(f.stem)]
        dwd_hornsea_df = pd.concat(
            (load_weather_data(f, "hornsea")
             for _, f in enumerate(dwd_hornsea_files)),
            ignore_index=True
        )
        dwd_hornsea_df.to_parquet(
            f"{self.config.root_dir}/dwd_hornsea_processed.parquet"
        )
        logger.info("Cleaned dwd hornsea data: file safed under {}",
                    self.config.root_dir)

        dwd_solar_rx = re.compile("dwd_icon_eu_pes10")
        dwd_solar_files = [f for f in weather_files if dwd_solar_rx.match(f.stem)]
        dwd_solar_df = pd.concat(
            (load_weather_data(f, "solar")
             for _, f in enumerate(dwd_solar_files)),
            ignore_index=True
        )
        dwd_solar_df.to_parquet(f"{self.config.root_dir}/dwd_solar_processed.parquet")
        logger.info("Cleaned dwd solar data: file safed under {}", self.config.root_dir)

        return dwd_hornsea_df, dwd_solar_df

    def merge_data(self, energy, hornsea, solar) -> None:
        logger.info("Start merging energy and weather data")

        merged_table = (
            hornsea
            .merge(solar, how="outer", on=["reference_time", "valid_time"])
            .merge(energy, how="inner", left_on="valid_time", right_on="dtm")
            .rename(columns={"Temperature_x": "temp_hornsea",
                             "Temperature_y": "temp_solar",
                             "hours_after_x": "hours_after"})
            .drop(columns="hours_after_y", axis=1)
        )

        merged_table.to_parquet(f"{self.config.root_dir}/merged_data.parquet")
        logger.info("Merged energy and weather data: file safed under {}",
                    self.config.root_dir)

        return merged_table

    def splitting_data(self, df_full):
        logger.info("Start splitting data in training and test data set")

        os.makedirs(self.config.training_data_path, exist_ok=True)
        logger.info("created directory at: {}", self.config.training_data_path)

        os.makedirs(self.config.test_data_path, exist_ok=True)
        logger.info("created directory at: {}", self.config.test_data_path)

        df_train = df_full.loc[df_full.reference_time < "2023-05-20"]
        df_test = df_full.loc[df_full.reference_time >= "2023-05-20"]

        label_wind: Final = "Wind_MWh_credit"  # mglw. in config-Datein schreiben
        featues_wind: list = ["RelativeHumidity", "temp_hornsea", "WindDirection",
                              "WindDirection:100", "WindSpeed", "WindSpeed:100",
                              "hours_after"]
        label_solar: Final = "Solar_MWh_credit"  # mglw. in config-Datein schreiben
        featues_solar: list = ["CloudCover", "SolarDownwardRadiation", "temp_solar",
                               "hours_after"]

        index_wind_train = df_train[df_train[label_wind].isna()].index
        index_solar_train = df_train[df_train[label_solar].isna()].index
        index_wind_test = df_test[df_test[label_wind].isna()].index
        index_solar_test = df_test[df_test[label_solar].isna()].index

        x_wind_train = df_train.drop(index_wind_train)[featues_wind]
        x_wind_train.to_parquet(
            f"{self.config.training_data_path}/x_wind_train.parquet"
        )
        x_wind_test = df_test.drop(index_wind_test)[featues_wind]
        x_wind_test.to_parquet(f"{self.config.test_data_path}/x_wind_test.parquet")
        x_solar_train = df_train.drop(index_solar_train)[featues_solar]
        x_solar_train.to_parquet(
            f"{self.config.training_data_path}/x_solar_train.parquet"
        )
        x_solar_test = df_test.drop(index_solar_test)[featues_solar]
        x_solar_test.to_parquet(f"{self.config.test_data_path}/x_solar_test.parquet")
        y_wind_train = df_train.drop(index_wind_train)[label_wind]
        y_wind_train.to_frame()\
            .to_parquet(f"{self.config.training_data_path}/y_wind_train.parquet")
        y_wind_test = df_test.drop(index_wind_test)[label_wind]
        y_wind_test.to_frame()\
            .to_parquet(f"{self.config.test_data_path}/y_wind_test.parquet")
        y_solar_train = df_train.drop(index_solar_train)[label_solar]
        y_solar_train.to_frame()\
            .to_parquet(f"{self.config.training_data_path}/y_solar_train.parquet")
        y_solar_test = df_test.drop(index_solar_test)[label_solar]
        y_solar_test.to_frame()\
            .to_parquet(f"{self.config.test_data_path}/y_solar_test.parquet")

    def transform_data(self) -> None:
        logger.info("Start transforming (feature engineering) the data")

        x_wind_train = pd.read_parquet(
            f"{self.config.training_data_path}/x_wind_train.parquet"
        )
        x_wind_test = pd.read_parquet(
            f"{self.config.test_data_path}/x_wind_test.parquet"
        )

        windspeed_train_pca = x_wind_train[["WindSpeed", "WindSpeed:100"]].to_numpy()
        windspeed_test_pca = x_wind_test[["WindSpeed", "WindSpeed:100"]].to_numpy()

        scale_pca_pipe = Pipeline([
            ("scaling", StandardScaler()),
            ("pca", PCA(n_components=1))
        ])

        windspeed_train_pca = scale_pca_pipe.fit_transform(windspeed_train_pca)
        windspeed_test_pca = scale_pca_pipe.transform(windspeed_test_pca)

        x_wind_train["WindSpeedPCA"] = windspeed_train_pca
        x_wind_test["WindSpeedPCA"] = windspeed_test_pca

        # transformed_df = df.assign(
        #     year=df["valid_time"].dt.year,
        #     month=df["valid_time"].dt.month,
        #     day=df["valid_time"].dt.day,
        #     hour=df["valid_time"].dt.hour
        # )

        # transformed_df.to_parquet(f"{self.config.root_dir}/transformed_data.parquet")
        # logger.info("Transformed data: file safed under {}",
        #             self.config.root_dir)
        # return transformed_df
