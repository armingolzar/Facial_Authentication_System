import tensorflow as tf
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


class ContrastiveLoss(tf.keras.losses.Loss):

    def __init__(self, margin=1.0, name="contrastive_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin

    
    def call(self, y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        squared_pred = tf.square(y_pred)
        margin_term = tf.square(tf.maximum(self.margin - y_pred, 0.0))
        cont_loss = tf.reduce_mean(y_true * squared_pred + (1 - y_true) * margin_term)

        return cont_loss
    

    def get_config(self):
        config = super().get_config()
        config.update({"margin" : self.margin})
        return config
    


def ploting_learning_curves(history, path):

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], color="red", label="Training Loss")
    plt.plot(history.history["val_loss"], color="green", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 

