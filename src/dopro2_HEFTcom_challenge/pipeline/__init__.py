"""Module for the different pipeline stages."""

from dopro2_HEFTcom_challenge.pipeline.stage_01_data_ingestion import \
    DataIngestionTrainingPipeline
from dopro2_HEFTcom_challenge.pipeline.stage_02_data_preparation import \
    DataPreparationTrainingPipeline
from dopro2_HEFTcom_challenge.pipeline.stage_03_training import ModelTrainingPipeline


__all__: list[str] = [
    "DataIngestionTrainingPipeline",
    "DataPreparationTrainingPipeline",
    "ModelTrainingPipeline"
]
