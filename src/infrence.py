import tensorflow as tf
import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.utils import euclidean_distance
import os 
from src.model import embedding_extractor
import src.config as config


verification_dev_path = ".\\data\\verification\\verification_dev.csv"
MODEL_CLASSIFICATION_PATH = ".\\models\\siamese_face_v1_classification.h5"
MODEL_CONTRASTIVE_PATH = ".\\models\\siamese_face_v1_contrastive.h5"


def model_arch(mode="contrastive"):

    if mode == "contrastive":
        inputA = tf.keras.layers.Input(config.IMAGE_SHAPE, name="inputA")
        inputB = tf.keras.layers.Input(config.IMAGE_SHAPE, name="inputB")

        embedding_extractor_model = embedding_extractor(config.IMAGE_SHAPE)
        embeddedA = embedding_extractor_model(inputA)
        embeddedB = embedding_extractor_model(inputB)

        distance = tf.keras.layers.Lambda(euclidean_distance)([embeddedA, embeddedB])
        # output = Dense(1, activation="sigmoid")(distance)
        siamese_face_model = tf.keras.models.Model(inputs=[inputA, inputB], outputs=distance)
        siamese_face_model.load_weights(MODEL_CONTRASTIVE_PATH)

    elif mode == "classification":
        inputA = tf.keras.layers.Input(config.IMAGE_SHAPE, name="inputA")
        inputB = tf.keras.layers.Input(config.IMAGE_SHAPE, name="inputB")

        embedding_extractor_model = embedding_extractor(config.IMAGE_SHAPE)
        embeddedA = embedding_extractor_model(inputA)
        embeddedB = embedding_extractor_model(inputB)

        distance = tf.keras.layers.Lambda(euclidean_distance)([embeddedA, embeddedB])
        output = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
        siamese_face_model = tf.keras.models.Model(inputs=[inputA, inputB], outputs=output)
        siamese_face_model.load_weights(MODEL_CLASSIFICATION_PATH)

    else:
        siamese_face_model = None
        print("Wrong mode!")

    return siamese_face_model


def process_to_image(img_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    return img











def face_authentication_system(data_path, mode="contrastive", threshold=0.5):

    data = pd.read_csv(data_path)
    siamese_model = model_arch(mode)
    path_imageA = data.iloc[:,  0].values
    path_imageB = data.iloc[:,  1].values
    label = data.iloc[:, -1].values










path_imageA, path_imageB, label = face_authentication_system(verification_dev_path)

















