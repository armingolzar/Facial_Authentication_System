import os

IMAGE_SHAPE = (224, 224, 3)

BATCH_SIZE = 64

EPOCHS = 50

DATA_TRAIN_PATH = ".\\data\\classification\\train1000\\*\\*.*"

DATA_TEST_PATH = ".\\data\\classification\\test1000\\*\\*.*"

MODELS_PATH = os.path.join(".", "models", "siamese_face_v1_contrastive.h5")

PLOTS_PATH = os.path.join(".", "assets", "learning_plot_contrastive.png")
