"""Functions that are often used in the process."""

from datetime import datetime
from pathlib import Path

from loguru import logger
import pandas as pd

from statsmodels.base.model import Results
from statsmodels.regression.quantile_regression import QuantRegResults


__all__: list[str] = [
    "day_ahead_market_times",
    "load_models",
    "prep_submission_in_json_format"
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
            "timestamp": submission_data["datetime"][i].isoformat(),
            "market_bid": submission_data["market_bid"][i],
            "probabilistic_forecast": {
                10: submission_data["q10"][i],
                20: submission_data["q20"][i],
                30: submission_data["q30"][i],
                40: submission_data["q40"][i],
                50: submission_data["q50"][i],
                60: submission_data["q60"][i],
                70: submission_data["q70"][i],
                80: submission_data["q80"][i],
                90: submission_data["q90"][i],
            }
        })

    data = {
        "market_day": market_day.strftime("%Y-%m-%d"),
        "submission": submission
    }

    return data
