from data_loader import get_image_path_and_categories, full_epoch_data_generator, preprocess_image, create_dataset
from model import embedding_extractor
from utils import euclidean_distance, ploting_learning_curves
import config
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model


train_addrs, train_cats = get_image_path_and_categories(config.DATA_TRAIN_PATH)
test_addrs, test_cats = get_image_path_and_categories(config.DATA_TEST_PATH)

steps_per_epoch = len(train_addrs) * 2 // config.BATCH_SIZE 

train_dataset = create_dataset(train_addrs, train_cats, config.BATCH_SIZE)
test_dataset = create_dataset(test_addrs, test_cats, config.BATCH_SIZE)

inputA = Input(config.IMAGE_SHAPE, name="inputA")
inputB = Input(config.IMAGE_SHAPE, name="inputB")

embedding_extractor_model = embedding_extractor(config.IMAGE_SHAPE)
embeddedA = embedding_extractor_model(inputA)
embeddedB = embedding_extractor_model(inputB)

distance = Lambda(euclidean_distance)([embeddedA, embeddedB])
output = Dense(1, activation="sigmoid")(distance)
siamese_face_model = Model(inputs=[inputA, inputB], outputs=output)

print("[INFO] Compilling the siamese_face_model...")
siamese_face_model.compile(
                            loss = "binary_crossentropy",
                            optimizer = "Adam",
                            metrics = ["accuracy"]
)

print("[INFO] Training the siamese_face_model...")
history = siamese_face_model.fit(train_dataset, validation_data=test_dataset,
                                  epochs=config.EPOCHS, steps_per_epoch=steps_per_epoch, 
                                  validation_steps=(len(test_addrs)*2)//config.BATCH_SIZE)

print("[INFO] Saving siamese_face_model...")
siamese_face_model.save(config.MODELS_PATH)

print("[INFO] Saving plot....")
ploting_learning_curves(history, config.PLOTS_PATH)



