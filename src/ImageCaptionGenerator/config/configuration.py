from src.ImageCaptionGenerator.constants import *
from src.ImageCaptionGenerator.utils.common import read_yaml, create_directory
from src.ImageCaptionGenerator.entity.config import (
    DataIngestionConfig,
    ModelTrainingConfig,
    TrainingConfig,
)


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directory([self.config.main_dir])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directory([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            kaggle_api=config.kaggle_api_key,
            local_files_dir=config.local_files_dir,
            unzip_dir=config.unzip_dir,
            batch_size=self.params.BATCH_SIZE,
        )

        return data_ingestion_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        model_config = self.config.model_training
        create_directory([model_config.model_dir])
        # create_directory([config.train_dir])

        model_training_config = ModelTrainingConfig(
            model_dir=model_config.model_dir,
            pkl_file=model_config.pkl_file,
            file_path=model_config.file_path,
            saved_model_dir=model_config.saved_model_dir,
            epochs=self.params.EPOCHS,
            batch_size=self.params.BATCH_SIZE,
            learning_rate=self.params.LEARNING_RATE,
        )

        return model_training_config

    def get_training_config(self) -> TrainingConfig:
        train_config = self.config.training
        create_directory([train_config.train_dir])
        # create_directory([config.train_dir])

        training_config = TrainingConfig(
            train_dir=train_config.train_dir,
            pkl_file=train_config.pkl_file,
            saved_trained_model_dir=train_config.saved_trained_model_dir,
            epochs=self.params.EPOCHS,
            batch_size=self.params.BATCH_SIZE,
        )

        return training_config
