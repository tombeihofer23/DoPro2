"""Data preparation component."""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import xarray as xr

from dopro2_HEFTcom_challenge.entity import DataPreparationConfig


class DataPreparation:
    """Class to performe data preparation."""

    def __init__(self, config: DataPreparationConfig) -> None:
        """
        Constructor for DataPreparation class.

        :param config: config values from config.yaml
        """

        self.config = config

    def cleaning_energy_data(self) -> None:
        # TODO: handling missing values, outliers, inconsistencies
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
        # TODO: Split into wind and solar
        df.to_parquet(f"{self.config.root_dir}/energy_processed.parquet")
        logger.info("Cleaned energy data: file safed under {}", self.config.root_dir)

    def cleaning_weather_data(self) -> None:
        logger.info("Start cleaning weather data")
        # weather_files = Path(self.config.weather_data_path).glob("*.nc")
        dwd_hornsea = xr.open_dataset("artifacts/raw_data/weather/dwd_icon_eu_hornsea_1_20200920_20231027.nc", engine="h5netcdf")
        dwd_hornsea_df = (
            dwd_hornsea["WindSpeed:100"]
            .mean(dim=["latitude", "longitude"])
            .to_dataframe()
            .reset_index()
        )
        dwd_hornsea_df = (
            dwd_hornsea_df
            .assign(ref_datetime=dwd_hornsea_df["ref_datetime"].dt.tz_localize("UTC"),
                    valid_datetime=(dwd_hornsea_df["ref_datetime"]
                    + pd.to_timedelta(dwd_hornsea_df["valid_datetime"], unit="hours"))
                    .dt.tz_localize("UTC")
                    )
        )
        dwd_hornsea_df.to_parquet(f"{self.config.root_dir}/dwd_hornsea_processed.parquet")
        logger.info("Cleaned dwd hornsea data: file safed under {}",
                    self.config.root_dir)

        dwd_solar = xr.open_dataset("artifacts/raw_data/weather/dwd_icon_eu_pes10_20200920_20231027.nc", engine="h5netcdf")
        dwd_solar_df = (
            dwd_solar["SolarDownwardRadiation"]
            .mean(dim="point")
            .to_dataframe()
            .reset_index()
        )
        dwd_solar_df = (
            dwd_solar_df
            .assign(ref_datetime=dwd_solar_df["ref_datetime"].dt.tz_localize("UTC"),
                    valid_datetime=(dwd_solar_df["ref_datetime"]
                    + pd.to_timedelta(dwd_solar_df["valid_datetime"], unit="hours"))
                    .dt.tz_localize("UTC")
                    )
        )
        dwd_solar_df.to_parquet(f"{self.config.root_dir}/dwd_solar_processed.parquet")
        logger.info("Cleaned dwd solar data: file safed under {}", self.config.root_dir)

    def merge_data(self) -> None:
        logger.info("Start merging energy and weather data")
        processed_files = Path(self.config.root_dir).glob("*.parquet")
        dfs = []
        for file in processed_files:
            df = pd.read_parquet(file)
            dfs.append(df)
        hornsea, solar, energy = dfs
        merged_table = (
            hornsea
            .merge(solar, how="outer", on=["ref_datetime", "valid_datetime"])
            .set_index("valid_datetime")
            .groupby("ref_datetime")
            .resample("30T")
            .interpolate("linear")
            .drop(columns="ref_datetime", axis=1)
            .reset_index()
            .merge(energy, how="inner", left_on="valid_datetime", right_on="dtm")
        )
        merged_table = merged_table[merged_table["valid_datetime"]
                                    - merged_table["ref_datetime"]
                                    < np.timedelta64(50, "h")]
        merged_table.rename(columns={"WindSpeed:100": "WindSpeed"}, inplace=True)
        merged_table.to_parquet(f"{self.config.root_dir}/merged_data.parquet")
        logger.info("Merged energy and weather data: file safed under {}",
                    self.config.root_dir)

    def transform_data(self) -> None:
        # TODO: feature scaling, encoding, ...
        logger.info("Start transforming data for modell training")
        merged_data = pd.read_parquet("artifacts/prepared_data/merged_data.parquet")
        model_data = merged_data[merged_data["SolarDownwardRadiation"].notnull()]
        model_data = merged_data[merged_data["WindSpeed"].notnull()]
        model_data["total_generation_MWh"] = model_data["Wind_MWh_credit"]\
            + model_data["Solar_MWh_credit"]
        model_data.to_parquet(f"{self.config.root_dir}/model_data.parquet")
        logger.info("Data ready to train the model: file safed under {}",
                    self.config.root_dir)

    # def reduce_data(self):

    # def splitting_data(self):
    #     ...
    #     # TODO: training, validation, test sets
