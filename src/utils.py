import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# array1 = np.array([1, 2, 3, 4, 5, 6])
# array2 = np.array([1, 2, 6, 12, 3, 9])

# distance_cpu = np.sqrt(np.sum(np.square(array1 - array2)))

# print(distance_cpu)

# tensor1 = tf.constant([[1, 2, 3], [10, 20, 30]], dtype="float32")
# tensor2 = tf.constant([[1, 2, 6], [15, 18, 25]], dtype="float32")
# vec = (tensor1, tensor2)

def euclidean_distance(vectors):

    tensorA, tensorB = vectors
    return tf.math.reduce_euclidean_norm(tensorA - tensorB, axis=1, keepdims=True)


def learning_curves(history):

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], color="red", label="Training Loss")
    plt.plot(history.history["val_loss"], color="green", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show() 

