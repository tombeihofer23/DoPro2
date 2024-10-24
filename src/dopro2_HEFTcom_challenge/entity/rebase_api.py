# code mostly from https://github.com/jbrowell/HEFTcom24/blob/main/comp_utils.py

"""Rebase-API-Klasse."""

import os
from typing import Literal

from loguru import logger
import pandas as pd
import requests
from requests import Session

from dopro2_HEFTcom_challenge.utils import (
    day_ahead_market_times,
    load_weather_data,
    weather_df_to_xr
)


class RebaseAPI:
    """Rebase-API-Klasse zum abrufen und abgeben der Daten."""

    challenge_id = "heftcom2024"
    base_url = "https://api.rebase.energy"

    def __init__(
        self,
        api_key=None
    ):
        if api_key is None:
            self.api_key = os.environ.get("REBASE_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        self.session = Session()
        self.session.headers = self.headers  # type: ignore

    def get_variable(
        self,
        day: str,
        variable: Literal["market_index",
                          "day_ahead_price",
                          "imbalance_price",
                          "wind_total_production",
                          "solar_total_production",
                          "solar_and_wind_forecast"
                          ]
    ) -> pd.DataFrame:
        """
        GET-Request für Challange-Data

        :param day: Datum der abgefragten Daten im Format YYYY-MM-dd
        :param variable: Variable, die abgefragt werden soll
        :return: abgefragte Daten
        :rtype: DataFrame
        """
        url = f"{self.base_url}/challenges/data/{variable}"
        params = {"day": day}
        logger.debug("GET from {}, day={}", url, day)
        resp = self.session.get(url, params=params)
        logger.debug("statuscode={}", resp.status_code)

        data = resp.json()
        df = pd.DataFrame(data)
        return df

    def get_solar_wind_forecast(self, day: str) -> pd.DataFrame:
        """
        GET-Request für Solar/Wind-Vorhersagen

        :param day: Datum der abgefragten Daten
        :return: abgefragte Daten
        :rtype: DataFrame
        """
        url = f"{self.base_url}/challenges/data/solar_and_wind_forecast"
        params = {"day": day}
        logger.debug("GET from {}, day={}", url, day)
        resp = self.session.get(url, params=params)
        logger.debug("statuscode={}", resp.status_code)

        data = resp.json()
        df = pd.DataFrame(data)
        return df

    def get_day_ahead_demand_forecast(self) -> list:
        """
        GET-Request für vorhergesagten Bedarf des nächsten Tages

        :return: abgefragte Daten
        :rtype: Liste mit JSON-encodierten Responses
        """
        url = f"{self.base_url}/challenges/data/day_ahead_demand"
        logger.debug("GET from {}", url)
        resp = self.session.get(url)
        logger.debug("statuscode={}", resp.status_code)

        return resp.json()

    def get_margin_forecast(self) -> list:
        """
        GET-Request für vorhergesagte Marge

        :return: abgefragte Daten
        :rtype: Liste mit JSON-encodierten Responses
        """
        url = f"{self.base_url}/challenges/data/margin_forecast"
        logger.debug("GET from {}", url)
        resp = self.session.get(url)
        logger.debug("statuscode={}", resp.status_code)

        return resp.json()

    def query_weather_latest(
        self,
        model: Literal["DWD_ICON-EU", "NCEP_GFS"],
        lats: list[float],
        lons: list[float],
        variables: str,
        query_type: Literal["grid", "points"]
    ) -> list:
        """
        POST-Request zum abfragen der Wetterdaten

        :param model: genutztes Datenset
        :param lats: Liste der Breitengrade
        :param lons: Liste der Längengrade
        :param variables: Liste der Wettervariablen
        :param query_type: Format, wie Daten zurückgegeben werden
        :return: abgefragte Daten
        :rtype: Liste mit JSON-encodierten Responses
        """
        url = f"{self.base_url}/weather/v2/query"
        body = {
            "model": model,
            "latitude": lats,
            "longitude": lons,
            "variables": variables,
            "type": query_type,
            "output-format": "json",
            "forecast-horizon": "latest"
        }
        logger.debug("POST from {}, model={}, lat={}, long={}, "
                     "variables={}, type={}",
                     url, model, lats, lons, variables, query_type)
        resp = requests.post(
            url, json=body, headers={"Authorization": self.api_key}, timeout=180
        )
        logger.debug("statuscode={}", resp.status_code)

        return resp.json()

    def query_weather_latest_points(
        self,
        model: Literal["DWD_ICON-EU", "NCEP_GFS"],
        lats: list[float],
        lons: list[float],
        variables: str,
    ) -> pd.DataFrame:
        """ Daten werden als Liste zurückgegeben"""

        data = self.query_weather_latest(
            model, lats, lons, variables, "points"
        )

        df = pd.DataFrame()
        for i, _ in enumerate(data):
            new_df = pd.DataFrame(data[i])
            new_df["point"] = i
            new_df["latitude"] = lats[i]
            new_df["longitude"] = lons[i]
            df = pd.concat([df, new_df])

        return df

    def query_weather_latest_grid(
        self,
        model: Literal["DWD_ICON-EU", "NCEP_GFS"],
        lats: list[float],
        lons: list[float],
        variables: str,
    ) -> pd.DataFrame:
        """Daten werden 'eben' zurückgegeben"""

        data = self.query_weather_latest(model, lats, lons, variables, "grid")
        df = pd.DataFrame(data)

        return df

    def get_hornsea_dwd(self):
        """
        Wetterdaten bei Hornsea1-Koordinaten von DWD_ICON-EU
        als 6x6 Grid
        """

        lats = [53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
        lons = [1.702, 1.767, 1.832, 1.897, 1.962, 2.027]

        variables = "WindSpeed, WindSpeed:100, WindDirection, " \
                    "WindDirection:100, Temperature, RelativeHumidity"
        return self.query_weather_latest_grid(
            "DWD_ICON-EU", lats, lons, variables
        )

    def get_hornsea_gfs(self):
        """
        Wetterdaten bei Hornsea1-Koordinaten von NCEP_GFS
        als 3x3 Grid
        """

        lats = [53.59, 53.84, 54.09]
        lons = [1.522, 1.772, 2.022]

        variables = "WindSpeed, WindSpeed:100, WindDirection, " \
                    "WindDirection:100, Temperature, RelativeHumidity"
        return self.query_weather_latest_grid(
            "NCEP_GFS", lats, lons, variables
        )

    def get_pes10_nwp(self, model: Literal["DWD_ICON-EU", "NCEP_GFS"]):
        """Wetterdaten für Solar"""

        lats = [52.4872562, 52.8776682, 52.1354277, 52.4880497, 51.9563696,
                52.2499177, 52.6416477, 52.2700912, 52.1960768, 52.7082618,
                52.4043468, 52.0679429, 52.024023, 52.7681276, 51.8750506,
                52.5582373, 52.4478922, 52.5214863, 52.8776682, 52.0780721]
        lons = [0.4012455, 0.7906532, -0.2640343, -0.1267052, 0.6588173,
                1.3894081, 1.3509559, 0.7082557, 0.1534462, 0.7302284,
                1.0762977, 1.1751747, 0.2962684, 0.1699257, 0.9115028,
                0.7137489, 0.1204872, 1.5706825, 1.1916542, -0.0113488]

        variables = "SolarDownwardRadiation, CloudCover, Temperature"
        return self.query_weather_latest_points(model, lats, lons, variables)

    def get_demand_nwp(self, model: Literal["DWD_ICON-EU", "NCEP_GFS"]):
        """Wetterdaten für Bedarf"""

        lats = [51.479, 51.453, 52.449, 53.175, 55.86, 53.875, 54.297]
        lons = [-0.451, -2.6, -1.926, -2.986, -4.264, -0.442, -1.533]

        variables = "Temperature, WindSpeed, WindDirection, " \
                    "TotalPrecipitation, RelativeHumidity"
        return self.query_weather_latest_points(model, lats, lons, variables)

    def get_latest_forecast_data(self) -> pd.DataFrame:
        """
        Load lates data from rebase api and puts it in the right
        form for prediction.

        :return: data in the correct form for the model
        :rtype: DataFrame
        """
        raw_hornsea = self.get_hornsea_dwd()
        hornsea_df = load_weather_data(weather_df_to_xr(raw_hornsea),
                                       dtype="hornsea", api=True)

        raw_solar = self.get_pes10_nwp("DWD_ICON-EU")
        solar_df = load_weather_data(weather_df_to_xr(raw_solar),
                                     dtype="solar", api=True)

        latest_forecast_df = (
            hornsea_df
            .merge(solar_df, how="outer", on=["reference_time", "valid_time"])
            .rename(columns={"Temperature_x": "temp_hornsea",
                             "Temperature_y": "temp_solar",
                             "hours_after_x": "hours_after"})
            .drop(columns=["hours_after_y", "latitude", "longitude"], axis=1)
            .set_index("valid_time")
        )
        day_ahead_market_times_df = day_ahead_market_times()
        latest_forecast_df = (
            latest_forecast_df
            .loc[day_ahead_market_times_df]
            .reset_index(names="valid_time")
        )
        # latest_forecast_df["year"] = latest_forecast_df.valid_time.dt.year
        # latest_forecast_df["month"] = latest_forecast_df.valid_time.dt.month
        # latest_forecast_df["day"] = latest_forecast_df.valid_time.dt.day
        # latest_forecast_df["hour"] = latest_forecast_df.valid_time.dt.hour

        columns = ["valid_time", "hours_after", "CloudCover",
                   "SolarDownwardRadiation", "temp_hornsea",
                   "RelativeHumidity", "temp_solar", "WindDirection",
                   "WindDirection:100", "WindSpeed", "WindSpeed:100"]
                #    "year", "month", "day", "hour"]
        latest_forecast_df = latest_forecast_df.dropna()[columns]

        return latest_forecast_df

    def submit(self, data) -> None:
        """Ergebnisse zur API schicken."""

        url = f"{self.base_url}/challenges/{self.challenge_id}/submit"

        resp = self.session.post(url, headers=self.headers, json=data)
        logger.info(resp)
        logger.info(resp.text)
