import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow import keras

from Util.logger import get_datasets

datasets = get_datasets()

# TODO: Parse raw data.
# TODO: Standard Scaler.
for idx, dataset in enumerate(datasets):

    X, y = dataset


    # Create model using sequntial API
    model = keras.models.Sequential()



    # Add a Dense hidden layer with 6 neurons.
    # It uses the ReLU activation function.
    model.add(keras.layers.Dense(6, activation="relu"))

    # Add a second Dense hidden layer with 100 neurons.
    model.add(keras.layers.Dense(100, activation="relu"))

    # Add dense outputlayer with 3 neurons (one for each class).
    # Use the softmax activation function because the classes are exclusive.
    model.add(keras.layers.Dense(3, activation="softmax"))

    # # Print a summary of the model.
    # print(model.summary())


    # Compile model and specify loss function + optimiser.
    # Additionally specify extra metrics to compute.

    sgd = SGD(lr=0.001)

    model.compile(loss="categorical_crossentropy",
                  optimizer=sgd,
                  metrics=["accuracy"])

    y = to_categorical(y)

    # Train and evaluate model.
    history = model.fit(X, y, epochs=300)


    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    print("\n")
