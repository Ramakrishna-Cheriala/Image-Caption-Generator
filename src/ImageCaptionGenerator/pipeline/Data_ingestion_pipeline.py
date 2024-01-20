from src.ImageCaptionGenerator.components.Data_ingestion import DataIngestion
from src.ImageCaptionGenerator.config.configuration import ConfigurationManager
import os


feature_path = os.path.join("artifact", "feature.p")


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        # data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        # data_ingestion.data_splitting()
        if os.path.exists(feature_path):
            data_ingestion.image_features()
        else:
            print("Feaure extraction already done!!")
