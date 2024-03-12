import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64')
from Util.deep_logger import load_deep_data
from tensorflow import keras

from Util.deep_skfold import deep_cv_score


def lstm_closure():
    def build_lstm():
        model = keras.models.Sequential([
            keras.layers.LSTM(64, input_shape=(2500, 3)),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(3, activation="softmax")
        ])

        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model

    return build_lstm


def main():
    d1X, d1y, d2X, d2y, d3X, d3y = load_deep_data()

    accuracies = deep_cv_score(d2X, d2y, lstm_closure(), batch_size=1)

    print(f"Deep CV scores for the LSTM model: {accuracies}")



