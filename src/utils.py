import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# array1 = np.array([1, 2, 3, 4, 5, 6])
# array2 = np.array([1, 2, 6, 12, 3, 9])

# distance_cpu = np.sqrt(np.sum(np.square(array1 - array2)))

# print(distance_cpu)

# tensor1 = tf.constant([[1, 2, 3, 4, 5, 6]], dtype="float32")
# tensor2 = tf.constant([[1, 2, 6, 12, 3, 9]], dtype="float32")

def euclidean_distance(tensorA, tensorB):
    return tf.sqrt(tf.reduce_sum(tf.square(tensorA - tensorB))).numpy()

# distance = euclidean_distance(array1, array2)
# print(distance)

