from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense

from Util.deep_logger import load_deep_data
from tensorflow import keras

from Util.deep_skfold import deep_cv_score


# TODO: Considder using separable conv1D.
# TODO: Check the default padding.
# TODO: Check if the kernel should be set to default values.
# TODO: Check regularisers.

def cnn_closure(learning_rate=1e-4, activation="relu", kernel_initializer="he_normal", n_hidden=0, dropout=False,
                n_filters=128):
    def build_cnn(learning_rate=learning_rate, activation=activation, kernel_initializer=kernel_initializer):
        model = keras.models.Sequential()
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # Create 1 Dimensional convolution layers.
        model.add(Conv1D(filters=n_filters, kernel_size=3, activation=activation, input_shape=(2500, 3),
                         kernel_initializer=kernel_initializer))
        for layer in range(n_hidden):
            model.add(
                Conv1D(filters=n_filters, kernel_size=3, activation=activation, kernel_initializer=kernel_initializer))

        # Dropout for regularization - should test without dropout.
        if dropout:
            model.add(Dropout(0.5))

        # Max pooling reduces the learned features to 1/4 of the size
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation=activation, kernel_initializer=kernel_initializer))
        model.add(Dense(3, activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        return model

    return build_cnn


# Fully convolutional architecture.
def cnn_closure2(learning_rate=1e-4, activation="relu", kernel_initializer="he_normal", n_hidden=0, dropout=False,
                 n_filters=64, input_shape=(2500, 3)):
    def build_cnn2(learning_rate=learning_rate, activation=activation, kernel_initializer=kernel_initializer):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=n_filters, kernel_size=3, padding="same", input_shape=(2500, 3), kernel_initializer=kernel_initializer)(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=n_filters, kernel_size=3, padding="same", kernel_initializer=kernel_initializer)(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=n_filters, kernel_size=3, padding="same", kernel_initializer=kernel_initializer)(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAvgPool1D()(conv3)
        output_layer = keras.layers.Dense(3, activation="softmax")(gap)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy", "sparse_categorical_accuracy"])
        return model

    return build_cnn2



# WaveNet
def wavenet_closure(kernel_initializer="he_normal"):
    def build_wavenet():
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[2500, 3]))

        for rate in (1, 2, 4, 8) * 2:
            model.add(keras.layers.Conv1D(filters=20, kernel_size=3, padding="causal", activation="relu", dilation_rate=rate, kernel_initializer=kernel_initializer))

        model.add(keras.layers.Conv1D(filters=10, kernel_size=1))

        model.add(Flatten())
        model.add(Dense(3, activation="softmax"))

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])


        return model

    return build_wavenet




def main():
    d1X, d1y, d2X, d2y, d3X, d3y = load_deep_data()

    accuracies = deep_cv_score(d2X, d2y, cnn_closure(), batch_size=16, epochs=50)

    print(f"Deep CV scores for the 1d-CNN model: {accuracies}")


if __name__ == '__main__':
    main()
