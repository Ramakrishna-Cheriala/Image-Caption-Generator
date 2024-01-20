from src.ImageCaptionGenerator.components.Model_training import Model_training, Training
from src.ImageCaptionGenerator.config.configuration import ConfigurationManager


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_preparation_config = config.get_model_training_config()
        training_config = config.get_training_config()
        model_preparation = Model_training(model_config=model_preparation_config)
        training_preparation = Training(
            train_config=training_config, model_config=model_preparation_config
        )
        model_preparation.preprocessing_captions()
        model_preparation.model_preparation()
        training_preparation.load_model()
        training_preparation.training_model()
