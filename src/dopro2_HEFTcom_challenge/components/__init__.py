"""Module for the components."""

from dopro2_HEFTcom_challenge.components.data_ingestion import DataIngestion
from dopro2_HEFTcom_challenge.components.data_preparation import DataPreparation


__all__: list[str] = [
    "DataIngestion",
    "DataPreparation"
]
