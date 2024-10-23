"""Functions that are often used in the process."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger
import pandas as pd
import xarray as xr

from statsmodels.base.model import Results
from statsmodels.regression.quantile_regression import QuantRegResults


__all__: list[str] = [
    "categorize_wind_dir",
    "get_season",
    "get_time_of_day",
    "day_ahead_market_times",
    "load_models",
    "load_weather_data",
    "prep_submission_in_json_format",
    "weather_df_to_xr"
]


def load_models(path: Path) -> list[Results]:
    """
    Load all quantile regression models (q10, ..., q90)

    :param path: Path to the model dictionary
    :return: List with all models
    :rtype: list[Results]
    """

    model_files = Path(path).glob("*.pickle")
    models = []
    for file in model_files:
        models.append(QuantRegResults.load(file))
    return models


def day_ahead_market_times() -> pd.DatetimeIndex:
    """
    creates a pandas DatatimeIndex
    with the correct timestamps, which handle the transition
    to and from daylighe saving.

    :return: correct timestamps for predictions
    :rtype: DatetimeIndex
    """

    today_date = pd.to_datetime("today")

    tomorrow_date = today_date + pd.Timedelta(1, unit="day")
    DA_Market = [pd.Timestamp(
        datetime(today_date.year, today_date.month, today_date.day, 23, 0, 0),
        tz="Europe/London"),
        pd.Timestamp(datetime(tomorrow_date.year, tomorrow_date.month,
                              tomorrow_date.day, 22, 30, 0),
                     tz="Europe/London")]

    DA_Market = pd.date_range(
        start=DA_Market[0], end=DA_Market[1],
        freq=pd.Timedelta(30, unit="minute")
    )

    return DA_Market


def prep_submission_in_json_format(submission_data: pd.DataFrame) -> dict:
    """
    Prepare submissions in JSON format.

    :param submission_data: DataFrame with the predicted values
    :return: submission data in JSON format
    :rtype: dict
    """

    market_day = pd.to_datetime("today") + pd.Timedelta(1, unit="day")
    submission = []

    if any(submission_data["market_bid"] < 0):
        submission_data.loc[submission_data["market_bid"] < 0, "market_bid"] = 0
        logger.warning("Some market bids were less than 0 and have been set to 0")

    if any(submission_data["market_bid"] > 1800):
        submission_data.loc[submission_data["market_bid"] > 1800, "market_bid"] = 1800
        logger.warning("Some market bids were greater than 1800"
                       "and have been set to 1800")

    for i in range(len(submission_data.index)):
        submission.append({
            "timestamp": submission_data["valid_time"][i].isoformat(),
            "market_bid": float(submission_data["market_bid"][i]),
            "probabilistic_forecast": {
                10: float(submission_data["q10"][i]),
                20: float(submission_data["q20"][i]),
                30: float(submission_data["q30"][i]),
                40: float(submission_data["q40"][i]),
                50: float(submission_data["q50"][i]),
                60: float(submission_data["q60"][i]),
                70: float(submission_data["q70"][i]),
                80: float(submission_data["q80"][i]),
                90: float(submission_data["q90"][i]),
            }
        })

    data = {
        "market_day": market_day.strftime("%Y-%m-%d"),
        "submission": submission
    }

    return data


def weather_df_to_xr(weather_data: pd.DataFrame) -> xr.Dataset:
    """
    Turns rebase api weather dataframe into xarray Dataset.

    :param weather_data: weather data from api call
    :return: weather data as xarray Dataset
    :rtype: Dataset
    """

    weather_data["ref_datetime"] = pd.to_datetime(weather_data["ref_datetime"],
                                                  utc=True)
    weather_data["valid_datetime"] = pd.to_datetime(weather_data["valid_datetime"],
                                                    utc=True)

    if "point" in weather_data.columns:
        weather_data = weather_data.set_index(["ref_datetime",
                                               "valid_datetime",
                                               "point"])
    else:
        weather_data = pd.melt(weather_data, id_vars=["ref_datetime", "valid_datetime"])

        weather_data = (
            pd.concat(
                [weather_data, weather_data["variable"].str.split("_", expand=True)],
                axis=1
            )
            .drop(["variable", 1, 3], axis=1)
            .rename(columns={0: "variable", 2: "latitude", 4: "longitude"})
            .set_index(["ref_datetime", "valid_datetime",
                        "longitude", "latitude"])
            .pivot(columns="variable", values="value")
        )

    weather_data = weather_data.to_xarray()  # type: ignore

    weather_data["ref_datetime"] = pd.DatetimeIndex(
        weather_data["ref_datetime"].values, tz="UTC"
    )
    weather_data["valid_datetime"] = pd.DatetimeIndex(
        weather_data["valid_datetime"].values, tz="UTC"
    )

    return weather_data  # type: ignore


def load_weather_data(
    input: xr.Dataset | Path,
    dtype: Literal["hornsea", "solar"],
    api: bool = False,
) -> pd.DataFrame:
    """
    Load xarray weather data, preprocess it and return an dataframe.

    :param dataset: xarray dataset with weather data
    :param dtype: wind (hornsea) or solar data
    :param api: True, if data comes from api call, else from a dataset
    :return: DataFrame with preprocessed data
    :rtype: DataFrame
    """

    if isinstance(input, Path):
        dataset = xr.open_dataset(input)
    else:
        dataset = input

    dimension = ["latitude", "longitude"] if dtype == "hornsea" else ["point"]
    df = (
        dataset
        .mean(dim=dimension)
        .to_dataframe()
        .reset_index()
    ).rename(columns={"ref_datetime": "reference_time", "valid_datetime": "valid_time"})
    if api:
        df["hours_after"] = ((df["valid_time"] - df["reference_time"])
                             .dt.total_seconds() // 3600).astype(int)
        df = (
            df
            .set_index("valid_time").groupby("reference_time")
            .resample("30min").interpolate("linear")
            .drop(columns="reference_time", axis=1)
            .reset_index()
        )
        return df
    df = (
        df.assign(
            reference_time=df["reference_time"].dt.tz_localize("UTC"),
            hours_after=df["valid_time"],
            valid_time=(
                df["reference_time"] + pd.to_timedelta(df["valid_time"], unit="hours")
            ).dt.tz_localize("UTC")
        ).set_index("valid_time").groupby("reference_time")
         .resample("30min").interpolate("linear")
         .drop(columns="reference_time", axis=1)
         .reset_index()
    )

    return df


def get_time_of_day(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "night"


def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def categorize_wind_dir(degree):
    if degree >= 337.5 or degree < 22.5:
        return "N"
    elif 22.5 <= degree < 67.5:
        return "NE"
    elif 67.5 <= degree < 112.5:
        return "E"
    elif 112.5 <= degree < 157.5:
        return "SE"
    elif 157.5 <= degree < 202.5:
        return "S"
    elif 202.5 <= degree < 247.5:
        return "SW"
    elif 247.5 <= degree < 292.5:
        return "W"
    else:
        return "NW"


def remove_upperbound(merged_table_features, percentage=0.02):
    columns = ['SolarDownwardRadiation', 'temp_hornsea', 'temp_solar', 'WindSpeed', 'WindSpeed:100']
    n = round(len(merged_table_features) * percentage)
    indexes = set()
    for col in columns:
        indexes.update(set(merged_table_features[col].nlargest(n).index))
    return merged_table_features.drop(indexes).reset_index(drop=True)
