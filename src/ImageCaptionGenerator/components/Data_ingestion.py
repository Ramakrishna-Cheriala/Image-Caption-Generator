import os
import urllib.request as request
import zipfile
from src.ImageCaptionGenerator import logger
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import image
import pandas as pd
import re
import csv
from glob import glob
from tensorflow.keras.models import Model, Sequential
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
from pickle import dump, load

from src.ImageCaptionGenerator.config.configuration import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            dataset_name = self.config.kaggle_api
            KaggleApi().dataset_download_files(
                dataset_name, path=self.config.local_files_dir, force=True
            )
            logger.info("File downloaded successfully from Kaggle.")
        except Exception as e:
            logger.error(f"Error downloading file from Kaggle: {e}")
            raise e

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        filepath = os.path.join("Data", "dataset.zip")
        p = os.path.join("artifact", "data_ingestion", "data")
        if os.path.exists(p):
            logger.info("files already exist")
        else:
            os.makedirs(unzip_path, exist_ok=True)
            logger.info("Extracting zip files............")
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info("Zip files extracted successfully......................")

    def image_features(self):
        # os.makedirs(self.config.local_files_dir,exist_ok=True)
        images_dir = os.path.join("artifact", "data_ingestion", "data", "Images/")
        save_path = os.path.join("artifact", "features.p")
        images = glob(images_dir + "*.jpg")
        print("Total images in dataset:", len(images))

        model = Xception(include_top=False, pooling="avg")
        features = {}
        for img in tqdm(os.listdir(images_dir)):
            filename = images_dir + "/" + img
            image = Image.open(filename)
            image = image.resize((299, 299))
            image = np.expand_dims(image, axis=0)
            # image = preprocess_input(image)
            image = image / 127.5
            image = image - 1.0

            feature = model.predict(image)
            features[img] = feature

        dump(features, open("features.p", "wb"))
