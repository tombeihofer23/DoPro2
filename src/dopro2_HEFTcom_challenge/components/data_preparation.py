"""Data preparation component."""

from pathlib import Path

from loguru import logger
# import numpy as np
import pandas as pd
import re
# import xarray as xr

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
        )

        merged_table.to_parquet(f"{self.config.root_dir}/merged_data.parquet")
        logger.info("Merged energy and weather data: file safed under {}",
                    self.config.root_dir)

        return merged_table

    def transform_data(self, df) -> None:
        ...

    # def reduce_data(self):

    # def splitting_data(self):
    #     ...
    #
