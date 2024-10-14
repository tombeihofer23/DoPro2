"""Fifth ML Pipeline stage: predict on new data."""

import pandas as pd
from xgboost import XGBRegressor

from dopro2_HEFTcom_challenge.utils import (
    prep_submission_in_json_format
)
from dopro2_HEFTcom_challenge.entity import RebaseAPI


class PredictionPipeline:
    """Pipeline that make predictions on you data from Rebase API."""

    api: RebaseAPI = RebaseAPI()

    def __init__(self):
        pass

    def predict(self) -> None:
        """Load model and latest forecasts to make prediction."""

        latest_data = self.api.get_latest_forecast_data()
        model = XGBRegressor()
        model.load_model(r"artifacts\training\models\total_14-10-24.json")

        predictions = model.predict(
            latest_data.drop(columns="valid_time", axis=1).to_numpy()
        )
        predictions_df = pd.DataFrame(predictions,
                                      columns=["q10", "q20", "q30", "q40", "q50",
                                               "q60", "q70", "q80", "q90"])

        submission_data = latest_data.join(predictions_df)
        submission_data["market_bid"] = submission_data["q50"]

        submission_data_json = prep_submission_in_json_format(submission_data)
        print(submission_data_json)

        self.api.submit(submission_data_json)


if __name__ == "__main__":
    obj = PredictionPipeline()
    obj.predict()
