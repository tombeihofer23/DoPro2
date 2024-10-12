"""Functions that are often used in the process."""

from pathlib import Path

from statsmodels.base.model import Results
from statsmodels.regression.quantile_regression import QuantRegResults


def load_models(path: Path) -> list[Results]:
    model_files = Path(path).glob("*.pickle")
    models = []
    for file in model_files:
        models.append(QuantRegResults.load(file))
    return models
