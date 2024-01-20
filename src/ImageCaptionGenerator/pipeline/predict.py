import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
from tqdm import tqdm

# print(os.getcwd())
os.chdir("../../../")
# print(os.getcwd())


def generate_description(model, tokenizer, photo, max_length):
    # Extract features using Xception model
    # model = keras.applications.Xception(include_top=False, pooling="avg")
    # photo_features = model.predict(photo)

    # # Print input shape for debugging
    print("Shape of photo_features:", photo.shape)

    in_text = "[start]"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Reshape the photo_features to match the expected input shape
        reshaped_photo_features = photo.reshape((1, -1))
        yhat = model.predict([reshaped_photo_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "end":
            break
    return in_text


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def load_image(filename):
    img = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    model = keras.applications.Xception(include_top=False, pooling="avg")
    feature = model.predict(img_array)

    return feature


def main():
    # Paths
    model_path = os.path.join("artifact", "trained_model", "model.h5")
    path = os.path.join("artifact", "captions_data.pkl")
    with open(path, "rb") as pkl:
        data = pickle.load(pkl)
    tokenizer = data["tokenizer"]
    image_path = os.path.join("667626_18933d713e.jpg")

    # Load the model
    model = load_model(model_path)

    photo = load_image(image_path)

    # Generate description
    description = generate_description(model, tokenizer, photo, data["max_length"])
    print("Generated Description:", description)


if __name__ == "__main__":
    main()
