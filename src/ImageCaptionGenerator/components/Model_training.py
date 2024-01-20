import os
from src.ImageCaptionGenerator import logger
import urllib.request as request
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import re
import csv
from glob import glob
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dropout, Embedding, Masking
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import (
    LSTM,
    Concatenate,
    BatchNormalization,
    Bidirectional,
    RepeatVector,
)
from tensorflow.keras.layers import Add
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16
import pickle
import cv2
import tensorflow as tf
from itertools import islice
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import math
from tensorflow.keras.callbacks import ModelCheckpoint
import json
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from src.ImageCaptionGenerator.config.configuration import (
    ModelTrainingConfig,
    TrainingConfig,
)


class Model_training:
    def __init__(self, model_config: ModelTrainingConfig):
        self.model_config = model_config

    def preprocessing_captions(self):
        caption_dict = {}
        caption_list = []
        c = 0
        path = os.path.join("artifact", "captions_data.pkl")

        with open(self.model_config.file_path, "r") as f:
            next(f)
            captions = f.read()

        for line in tqdm(captions.split("\n")):
            token = line.split(",")

            if len(line) < 2:
                continue

            image_id, caption = token[0], token[1:]
            # image_id = image_id.split('.')[0]
            caption = " ".join(caption)

            if image_id not in caption_dict:
                caption_dict[image_id] = []
            caption_dict[image_id].append(caption)

        logger.info(f"Length of caption_dict: {len(caption_dict)}")

        keys_list = list(caption_dict.keys())[:2]
        for key in keys_list:
            print(key, caption_dict[key])

        for k, v in caption_dict.items():
            for i in range(len(v)):
                sentence = v[i]
                sentence = sentence.lower()
                sentence = sentence.replace("[^A-Za-z]", "")
                sentence = sentence.replace("\s+", " ")
                words = sentence.split()
                words = [word for word in words if len(word) > 1]
                sentence = "[start] " + " ".join(words) + " [end]"
                v[i] = sentence

        # keys_list = list(caption_dict.keys())[:2]
        # for key in keys_list:
        #     print(key, caption_dict[key])

        for k in caption_dict:
            for caption in caption_dict[k]:
                caption_list.append(caption)

        # print(sentence_list[:10])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(caption_list)

        max_length = max(len(caption.split()) for caption in caption_list)
        vocab_size = len(tokenizer.word_index) + 1

        logger.info(f"Vocabulary Size: {vocab_size}, Max Length: {max_length}")

        # Saving Pickle file
        logger.info("Saving Pickle file.................")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "caption_dict": caption_dict,
                    "max_length": max_length,
                    "vocab_size": vocab_size,
                    "tokenizer": tokenizer,
                },
                f,
            )

    def model_preparation(self):
        logger.info("Model Preparation Started.............................")

        # Load preprocessed data
        path = os.path.join("artifact", "captions_data.pkl")
        with open(path, "rb") as pkl:
            data = pickle.load(pkl)

        max_length = data["max_length"]
        vocab_size = data["vocab_size"]
        print(vocab_size, "\n", max_length)

        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation="relu")(fe1)
        # LSTM sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        # Merging both models
        decoder1 = Add()([fe2, se3])
        decoder2 = Dense(256, activation="relu")(decoder1)
        outputs = Dense(vocab_size, activation="softmax")(decoder2)

        optimizer = Adam(learning_rate=0.001)  # Set your desired learning rate

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        plot_model(model, to_file="model_plot.png", show_shapes=True)
        img = plt.imread("model_plot.png")
        plt.imshow(img)
        plt.show()

        model.summary()

        logger.info("Model saving................................")
        self.save_model(path=self.model_config.saved_model_dir, model=model)

    def save_model(self, path: Path, model=tf.keras.Model):
        model.save(path)
        logger.info("Model Saved.............................")

    # features = image_features mapping = caption_dict


class Training:
    def __init__(self, train_config: TrainingConfig, model_config: ModelTrainingConfig):
        self.train_config = train_config
        self.model_config = model_config

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_config.saved_model_dir)

    def training_model(self):
        logger.info("Training Started.............................")

        # Load preprocessed data
        path = os.path.join("artifact", "captions_data.pkl")
        with open(path, "rb") as pkl:
            data = pickle.load(pkl)

        path1 = os.path.join("artifact", "features.p")
        with open(path1, "rb") as pkl:
            image_features = pickle.load(pkl)

        caption_dict = data["caption_dict"]
        max_length = data["max_length"]
        vocab_size = data["vocab_size"]
        tokenizer = data["tokenizer"]

        print(len(image_features), "\n", len(caption_dict), "\n", vocab_size)

        image_names = list(caption_dict.keys())
        split = int(len(image_names) * 0.80)
        train_data = image_names[:split]
        validation_data = image_names[split:]

        print(
            f"\nlength of train_data {len(train_data)}, length of validation_data {len(validation_data)}"
        )

        steps_per_epoch = (len(train_data) // self.train_config.batch_size) + 1
        validation_steps = len(validation_data) // self.train_config.batch_size + 1
        # steps = math.ceil(len(train_dict) / self.train_config.batch_size)
        logger.info(f"Number of steps per epoch: {steps_per_epoch}")

        if not isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer):
            raise TypeError("Loaded tokenizer is not an instance of Tokenizer class.")

        for i in range(self.train_config.epochs):
            print(i, "/", self.train_config.epochs, ":\n")
            train_generator = self.data_generator(
                caption_dict, image_features, tokenizer, max_length, vocab_size
            )
            self.model.fit_generator(
                train_generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1
            )
            self.model.save("models/model_" + str(i) + ".h5")

        self.save_model(
            path=self.train_config.saved_trained_model_dir, model=self.model
        )

    # create input-output sequence pairs from the image description.

    # data generator, used by model.fit_generator()
    def data_generator(
        self, train_data, image_features, tokenizer, max_length, vocab_size
    ):
        while 1:
            for key, description_list in train_data.items():
                # retrieve photo features
                feature = image_features[key][0]
                # print(description_list)
                input_image, input_sequence, output_word = self.create_sequences(
                    tokenizer, max_length, description_list, feature, vocab_size
                )
                yield [[input_image, input_sequence], output_word]

    def create_sequences(self, tokenizer, max_length, desc_list, feature, vocab_size):
        X1, X2, y = list(), list(), list()
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(feature)

                X2.append(in_seq)
                y.append(out_seq)

        return np.array(X1), np.array(X2), np.array(y)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
