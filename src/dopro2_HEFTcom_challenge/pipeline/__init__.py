"""Module for the different pipeline stages."""

from dopro2_HEFTcom_challenge.pipeline.stage_01_data_ingestion import \
    DataIngestionTrainingPipeline
from dopro2_HEFTcom_challenge.pipeline.stage_02_data_preparation import \
    DataPreparationTrainingPipeline
from dopro2_HEFTcom_challenge.pipeline.stage_03_training import ModelTrainingPipeline
from dopro2_HEFTcom_challenge.pipeline.stage_04_evaluation import \
    ModelEvaluationPipeline


__all__: list[str] = [
    "DataIngestionTrainingPipeline",
    "DataPreparationTrainingPipeline",
    "ModelEvaluationPipeline",
    "ModelTrainingPipeline"
]
