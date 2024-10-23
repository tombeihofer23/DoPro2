"""Data preparation component."""

import os
from pathlib import Path
from typing import Final
import numpy as np

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

        merged_table = merged_table.assign(
            year=merged_table["valid_time"].dt.year,
            month=merged_table["valid_time"].dt.month,
            day=merged_table["valid_time"].dt.day,
            hour=merged_table["valid_time"].dt.hour
        )

        # Der Zeitraum der Messungen endet am 19.05.2024 23:30 Uhr -> Alle Wettervorhersagen danach sind nicht relevant und können gedropped werden
        merged_table = merged_table.drop(merged_table[merged_table.valid_time >= "2024-05-20"].index).reset_index(drop=True)

        merged_table.to_parquet(f"{self.config.root_dir}/merged_data.parquet")
        logger.info("Merged energy and weather data: file safed under {}",
                    self.config.root_dir)
        return merged_table
      
    def create_features(self, merged_table):
        merged_table_features = merged_table
       
        # Berechnung der angepassten Sonnenstrahlung unter Berücksichtigung der Bewölkung
        merged_table_features['adjusted_solar_radiation'] = merged_table_features['SolarDownwardRadiation'] * (1 - merged_table_features['CloudCover'] / 100)

        # Interaktion zwischen Temperatur (x) und Sonnenstrahlung
        merged_table_features['temp_x_solar_interaction'] = merged_table_features['temp_hornsea'] * merged_table_features['SolarDownwardRadiation']
        # Interaktion zwischen Temperatur (y) und Sonnenstrahlung
        merged_table_features['temp_y_solar_interaction'] = merged_table_features['temp_solar'] * merged_table_features['SolarDownwardRadiation']

        # Berechnung der Windinteraktion unter Verwendung von Windgeschwindigkeit und -richtung
        merged_table_features['wind_interaction'] = merged_table_features['WindSpeed'] * np.cos(merged_table_features['WindDirection'])
        # Windinteraktion für Windgeschwindigkeit in 100 m Höhe
        merged_table_features['wind_interaction_100'] = merged_table_features['WindSpeed:100'] * np.cos(merged_table_features['WindDirection:100'])

        # Interaktion zwischen relativer Luftfeuchtigkeit und Windgeschwindigkeit
        merged_table_features['humidity_wind_interaction'] = merged_table_features['RelativeHumidity'] * merged_table_features['WindSpeed']

        # Gradient der Windgeschwindigkeit zwischen 100 m Höhe und Boden
        merged_table_features['wind_gradient'] = merged_table_features['WindSpeed:100'] * merged_table_features['WindSpeed']

        # Erstellen einer neuen Spalte für die zeitliche Verschiebung der Bewölkung um 1 Stunde
        merged_table_features['CloudCover_lag_1h'] = merged_table_features['CloudCover'].shift(1)
        # Berechnung der Änderung der Bewölkung im Vergleich zur vorherigen Stunde
        merged_table_features['cloud_cover_change'] = merged_table_features['CloudCover'] - merged_table_features['CloudCover_lag_1h']
        merged_table.to_parquet(f"{self.config.root_dir}/merged_data_features.parquet")
        logger.info("Created features: file safed under {}",
                    self.config.root_dir)
        return merged_table_features

    def splitting_data(self, df_full):
        logger.info("Start splitting data in training and test data set")

        os.makedirs(self.config.training_data_path, exist_ok=True)
        logger.info("created directory at: {}", self.config.training_data_path)

        os.makedirs(self.config.test_data_path, exist_ok=True)
        logger.info("created directory at: {}", self.config.test_data_path)

        df_train = df_full.loc[~df_full.reference_time.between(left="2023-09-01", right="2023-12-01", inclusive="left")]
        df_test = df_full.loc[df_full.reference_time.between(left="2023-09-01", right="2023-12-01", inclusive="left")]

        label_wind: Final = "Wind_MWh_credit"  # mglw. in config-Datein schreiben
        featues_wind: list = ["RelativeHumidity", "temp_hornsea", 'temp_solar', "WindDirection",
                              "WindDirection:100", "WindSpeed", "WindSpeed:100",
                              "year", "month", "day", "hour", "hours_after", 'wind_interaction',
                              'wind_interaction_100', 'humidity_wind_interaction', 'wind_gradient',
                              'CloudCover_lag_1h', 'cloud_cover_change']
        label_solar: Final = "Solar_MWh_credit"  # mglw. in config-Datein schreiben
        featues_solar: list = ["CloudCover", "SolarDownwardRadiation", "temp_hornsea",
                               "temp_solar", "year", "month", "day", "hour", "hours_after",
                               'adjusted_solar_radiation', 'temp_x_solar_interaction',
                               'temp_y_solar_interaction']
       
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

        logger.info("Transforming the wind data")

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

        x_wind_train.drop(columns=["WindSpeed", "WindSpeed:100"], axis=1, inplace=True)
        x_wind_test.drop(columns=["WindSpeed", "WindSpeed:100"], axis=1, inplace=True)

        x_wind_train.to_parquet(
            f"{self.config.training_data_path}/x_wind_train.parquet"
        )
        x_wind_test.to_parquet(f"{self.config.test_data_path}/x_wind_test.parquet")
        # transformed_df.to_parquet(f"{self.config.root_dir}/transformed_data.parquet")
        # logger.info("Transformed data: file safed under {}",
        #             self.config.root_dir)
        # return transformed_df