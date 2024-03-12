import keras
from keras import models
from keras.layers import Dense, Flatten, Activation, Conv1D, MaxPooling1D
from keras.utils import plot_model
from keras_visualizer import visualizer
from keras import layers
from Models.Deep_Models.fcnn import fcnn_closure
import visualkeras


def main():
    # FCNN
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(7500,)),
        layers.Dense(3, activation='softmax'),
    ])
    # model1 = models.Sequential()
    # model1.add(Dense(16, input_shape=(784,)))
    # model1.add(Dense(8))
    # model1.add(Dense(4))

    # visualizer(model, filename="fcnn", format='png', view=True)

    # CNN
    model = models.Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu", input_shape=(2500, 3),
                     kernel_initializer="he_normal"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(3, activation="softmax"))



    model = keras.models.Sequential([
        keras.layers.LSTM(64, input_shape=(2500, 3)),
        #keras.layers.BatchNormalization(),
        keras.layers.Dense(3, activation="softmax")
    ])
    plot_model(model, to_file="Images/lstm.png", show_shapes=True, show_layer_names=False,
               dpi=96,
               show_layer_activations=True,
               )

    # visualkeras.layered_view(model, to_file='cnn.png').show()


if __name__ == '__main__':
    main()
