from src.ImageCaptionGenerator.config.configuration import ConfigurationManager
from src.ImageCaptionGenerator.pipeline.Data_ingestion_pipeline import (
    DataIngestionPipeline,
)
from src.ImageCaptionGenerator.pipeline.Model_training_pipeline import (
    ModelTrainingPipeline,
)
from src.ImageCaptionGenerator import logger
import os


feature_path = os.path.join("artifact", "features.p")
model_path = os.path.join("artifact", "prepared_model", "model.h5")
trained_model_path = os.path.join("artifact", "trained_model", "model.h5")

try:
    STAGE_NAME = "Data Ingestion"
    if os.path.exists(feature_path):
        logger.info(f"{STAGE_NAME} already exists")

    else:
        logger.info(f"{STAGE_NAME} started")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"{STAGE_NAME} finished")

    STAGE_NAME = "Model Preparation and Training"
    if os.path.exists(model_path) and os.path.exists(trained_model_path):
        logger.info(f"{STAGE_NAME} already exists")
    else:
        logger.info(f"{STAGE_NAME} started")
        obj1 = ModelTrainingPipeline()
        obj1.main()
        logger.info(f"{STAGE_NAME} finished")


except Exception as e:
    logger.exception(e)
    raise e
