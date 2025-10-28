import tensorflow as tf
import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.utils import euclidean_distance
import os 
from src.model import embedding_extractor
import src.config as config


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
        siamese_face_model.load_weights(config.MODEL_CONTRASTIVE_PATH)

    elif mode == "classification":
        inputA = tf.keras.layers.Input(config.IMAGE_SHAPE, name="inputA")
        inputB = tf.keras.layers.Input(config.IMAGE_SHAPE, name="inputB")

        embedding_extractor_model = embedding_extractor(config.IMAGE_SHAPE)
        embeddedA = embedding_extractor_model(inputA)
        embeddedB = embedding_extractor_model(inputB)

        distance = tf.keras.layers.Lambda(euclidean_distance)([embeddedA, embeddedB])
        output = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
        siamese_face_model = tf.keras.models.Model(inputs=[inputA, inputB], outputs=output)
        siamese_face_model.load_weights(config.MODEL_CLASSIFICATION_PATH)

    else:
        siamese_face_model = None
        print("Wrong mode!")

    return siamese_face_model


def process_to_image(img_path):

    img = tf.io.read_file(os.path.join(".\\data\\verification", img_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    return img

def generator(imgA_path, imgB_path, labels):

    for imgA, imgB, label in zip(imgA_path, imgB_path, labels):
        yield(process_to_image(imgA), process_to_image(imgB), label)


def face_authentication_dataset(data_path):

    data = pd.read_csv(data_path)
    path_imageA = data.iloc[:,  0].values
    path_imageB = data.iloc[:,  1].values
    labels = data.iloc[:, -1].values

    dataset = tf.data.Dataset.from_generator(
                lambda : generator(path_imageA, path_imageB, labels), 
                output_signature=(
                    tf.TensorSpec(shape=config.IMAGE_SHAPE, dtype=tf.float32),
                    tf.TensorSpec(shape=config.IMAGE_SHAPE, dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
    )
    dataset = dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset



siamese_model = model_arch(mode="contrastive")

for thresh in config.THRESHOLD:
    print("thresh is :", thresh, "\n")
    y_true, y_pred = [], []
    dataset = face_authentication_dataset(config.CSV_TEST_PATH)
    for imgA_batch, imgB_batch, labels_batch in dataset:
        distance = siamese_model([imgA_batch, imgB_batch], training=False)
        preds_batch = tf.cast(distance < thresh, tf.int32)
        y_pred.extend(preds_batch.numpy().flatten())
        y_true.extend(labels_batch.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    print("y_true[:20]:", y_true[:20])
    print("y_pred[:20]:", y_pred[:20], "\n")

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("\n")

    print("Precision:", precision)
    print("Recall:", recall)
    print(f"End of {thresh} threshold", "\n")























