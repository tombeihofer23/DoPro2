"""Second ML Pipeline stage: data preparation."""

from typing import Final

from loguru import logger

from dopro2_HEFTcom_challenge.components import DataPreparation
from dopro2_HEFTcom_challenge.config import ConfigurationManager


STAGE_NAME: Final = "Data Preparation stage"


class DataPreparationTrainingPipeline:
    """Pipeline that prepare the data."""

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        energy = data_preparation.cleaning_energy_data()
        hornsea, solar = data_preparation.cleaning_weather_data()
        merged_data = data_preparation.merge_data(energy, hornsea, solar)
        merged_table_features = data_preparation.create_features(merged_data)
        data_preparation.splitting_data(merged_table_features)
        data_preparation.transform_data()


if __name__ == "__main__":
    try:
        logger.info(">>> stage {} started <<<", STAGE_NAME)
        obj = DataPreparationTrainingPipeline()
        obj.main()
        logger.info(">>> stage {} completed <<<", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
