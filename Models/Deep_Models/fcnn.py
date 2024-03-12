import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from Util.deep_logger import get_deep_datasets, load_deep_data
from Util.deep_skfold import deep_cv_score

# TODO: Change to He initialisation if using ReLU
RANDOM_SEARCH = False


def fcnn_closure(n_hidden=1, n_neurons=64, learning_rate=1e-4, activation="relu", kernel_initializer="he_normal"):
    def build_fcnn(n_hidden=n_hidden, n_neurons=n_neurons, learning_rate=learning_rate, input_shape=None):
        if input_shape is None:
            input_shape = [2500, 3]
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=input_shape))
        for layer in range(n_hidden):
            # Added He initialisation.
            model.add(keras.layers.Dense(n_neurons, activation=activation, kernel_initializer=kernel_initializer))
        model.add(keras.layers.Dense(3, activation="softmax"))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model

    return build_fcnn


def main():
    # datasets = get_deep_datasets()
    d1X, d1y, d2X, d2y, d3X, d3y = load_deep_data()
    datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]
    # TODO: Parse raw data.
    # TODO: Standard Scaler.
    for idx, dataset in enumerate(datasets):

        # TODO: Debug - only testing dataset 2 to begin with.
        if idx != 1:
            continue

        X, y = dataset

        # Create model using sequntial API

        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[2500, 3]),
        #     keras.layers.Dense(300, activation="relu"),
        #     keras.layers.Dense(300, activation="relu"),
        #     keras.layers.Dense(3, activation="softmax")
        # ])

        if RANDOM_SEARCH:
            build_fcnn = fcnn_closure()
            keras_classifier = keras.wrappers.scikit_learn.KerasClassifier(build_fcnn)

            # Search for hyperparameters.
            param_distribs = {
                "n_hidden": [1, 2, 3],
                "n_neurons": np.arange(1, 100),
                "learning_rate": reciprocal(3e-4, 3e-2),
            }

            rnd_search_cv = RandomizedSearchCV(keras_classifier, param_distribs, n_iter=10, cv=5)
            rnd_search_cv.fit(X, y, epochs=100, callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=3)])

            best_params = rnd_search_cv.best_params_
            print(f"Best Params: {best_params}")
            lr = best_params["learning_rate"]
            n_hidden = best_params["n_hidden"]
            n_neurons = best_params["n_neurons"]

            print(f"Best Score: {rnd_search_cv.best_score_}")

            # model = rnd_search_cv.best_estimator_.model

            # Train and evaluate model.
            # TODO: Perform CV
            accuracies = deep_cv_score(X, y, fcnn_closure(n_hidden=n_hidden, n_neurons=n_neurons, learning_rate=lr),
                                       n_splits=5,
                                       epochs=100)

        else:
            print("Performing Selu FCNN classification")
            accuracies = deep_cv_score(X, y,
                                       fcnn_closure(n_hidden=1, n_neurons=64, learning_rate=0.00039, activation="selu",
                                                    kernel_initializer="lecun_normal"), n_splits=10, epochs=100)

        print(accuracies)
