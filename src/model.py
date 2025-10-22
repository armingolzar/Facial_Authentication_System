from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Input, Dense
from tensorflow.keras.models import Model


def embedding_extractor(inputShape=(224, 224, 3), embeddingDim=48):

    input_layer = Input(inputShape, name = "input_layer")

    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(input_layer)
    maxpool1 = MaxPooling2D(name="maxpool1")(conv1)
    drop1 = Dropout(0.3, name="drop1")(maxpool1)


    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(drop1)
    maxpool2 = MaxPooling2D(name="maxpool2")(conv2)
    drop2 = Dropout(0.3, name="drop2")(maxpool2)


    conv3 = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3")(drop2)
    maxpool3 = MaxPooling2D(name="maxpool3")(conv3)
    
    globalaveragepool = GlobalAveragePooling2D(name="globalaveragepool")(maxpool3)

    output = Dense(embeddingDim, activation="linear", name="output")(globalaveragepool)

    model = Model(inputs=input_layer, outputs=output)

    model.summary()

    return model

