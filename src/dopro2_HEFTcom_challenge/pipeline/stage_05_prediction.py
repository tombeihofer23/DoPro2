"""Fifth ML Pipeline stage: predict on new data."""

from dopro2_HEFTcom_challenge.utils import (
    load_models,
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

        models = load_models("artifacts/training/models")  # type: ignore
        latest_data = self.api.get_latest_forecast_data()

        submission_data = latest_data.copy()
        for i, model in enumerate(models):
            submission_data[f"q{(i + 1) * 10}"] = model.predict(
                latest_data
            )
            submission_data.loc[submission_data[f"q{(i + 1) * 10}"] < 0,
                                f"q{(i + 1) * 10}"] = 0

        submission_data["market_bid"] = submission_data["q50"]

        submission_data_json = prep_submission_in_json_format(submission_data)
        print(submission_data_json)

        self.api.submit(submission_data_json)


if __name__ == "__main__":
    obj = PredictionPipeline()
    obj.predict()
