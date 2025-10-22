from data_loader import generating_pairs
from model import embedding_extractor
from utils import euclidean_distance, learning_curves
import config
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model



train_samples, train_labels = generating_pairs(config.DATA_TRAIN_PATH)
test_samples, test_labels = generating_pairs(config.DATA_TEST_PATH)
print(train_samples[0])
print(train_labels[0])
    
# inputA = Input(config.IMAGE_SHAPE, name="inputA")
# inputB = Input(config.IMAGE_SHAPE, name="inputB")

# embedding_extractor_model = embedding_extractor(config.IMAGE_SHAPE)
# embeddedA = embedding_extractor_model(inputA)
# embeddedB = embedding_extractor_model(inputB)

# distance = Lambda(euclidean_distance)([embeddedA, embeddedB])
# output = Dense(1, activation="sigmoid")(distance)
# siamese_face_model = Model(inputs=[inputA, inputB], outputs=output)




