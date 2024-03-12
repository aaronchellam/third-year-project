import numpy as np
from sklearn.preprocessing import StandardScaler

from Util.util import dict_to_df, get_analysis_df
from Util.deep_logger import load_deep_data, load_deep_train_sets
from Util.deep_skfold import deep_cv_score, TimingCallback
from fcnn import fcnn_closure
from lstm import lstm_closure
from cnn import cnn_closure
import pandas as pd
import tensorflow as tf
from timeit import default_timer as timer

data = load_deep_train_sets()

classifiers = {0: "FCNN", 1: "CNN"}

model_builders = [
    fcnn_closure(n_hidden=1, n_neurons=64, learning_rate=1e-4, activation="relu"),
    cnn_closure(learning_rate=1e-4)
]

for dataset_idx in range(3):
    X_trains, X_tests, y_trains, y_tests = data[dataset_idx]

    for model_idx, builder in enumerate(model_builders):
        accuracies = []
        counter = 1
        total_correct = 0
        total_training_time = 0

        for fold_idx in range(10):
            model = builder()
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            training_timer = TimingCallback()

            X_train = X_trains[fold_idx]
            X_test = X_tests[fold_idx]
            y_train = y_trains[fold_idx]
            y_test = y_tests[fold_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
                X_train.shape)
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

            model.fit(X_train, y_train, epochs=50, batch_size=16,
                      callbacks=[early_stopping, training_timer])

            y_pred_prob = model.predict(X_test)
            y_pred = [np.argmax(p_set) for p_set in y_pred_prob]
            n_correct = sum(y_pred == y_test)
            accuracy = n_correct / len(y_pred)
            accuracies.append(accuracy)

            train_time = sum(training_timer.logs)

            print(f"Predictions for fold {counter}:    {np.array(y_pred)}")
            print(f"Actual targets for fold {counter}: {y_test}")
            print(f"Accuracy for fold {counter}: {n_correct}/{len(y_pred)} = {accuracy}")

            print(f"The fold training time is {train_time} seconds")
            counter += 1
            total_correct += n_correct
            total_training_time += train_time

        #total_accuracy = total_correct / len(y)
        print(f"Accuracies across all folds:       {accuracies}")
        print(f"Average accuracy across all folds: {np.mean(accuracies)}")
        #print(f"Total accuracy across all folds: {total_correct}/{len(y)} = {total_accuracy}")
        print(f"The total training time across all folds: {total_training_time}")
        #print(f"The total number of model parameters is: {total_params}")
