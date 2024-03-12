import keras.callbacks
from keras.models import clone_model
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.base import clone
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
from colorama import init, Fore, Style

# TODO: Verify random seed works.
# TODO: Shuffle Data - I think skfold does this already.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.utils import class_weight


# TODO: Remove class weights if they perform worse!
# TODO: Remove scaling

def deep_cv_score(X, y, builder, n_splits=5, epochs=100, batch_size=1):
    global model
    tf.random.set_seed(42)
    np.random.seed(42)

    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True)
    accuracies = []
    counter = 1
    total_correct = 0
    total_training_time = 0

    # Compute class weights and format as a dictionary.
    # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    # class_weights = dict(enumerate(class_weights))
    #
    # #TODO: REMOVE!!!
    # class_weights[0] *= 3

    for train_index, test_index in skfolds.split(X, y):
        tf.keras.backend.clear_session()
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        training_timer = TimingCallback()
        print(f"\nPerforming DeepCV iteration {counter} of {n_splits}")

        model = builder()
        # sgd = SGD(lr=0.001)
        # model.compile(loss="sparse_categorical_crossentropy",
        #               optimizer="adam",
        #               metrics=["accuracy"])

        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]

        # TODO: STANDARD SCALING PRODUCES NEGATIVE FEATURE VALUES.
        scaler = StandardScaler()
        X_train_folds = scaler.fit_transform(X_train_folds.reshape(-1, X_train_folds.shape[-1])).reshape(
            X_train_folds.shape)
        X_test_fold = scaler.transform(X_test_fold.reshape(-1, X_test_fold.shape[-1])).reshape(X_test_fold.shape)

        model.fit(X_train_folds, y_train_folds, epochs=epochs, batch_size=batch_size,
                  callbacks=[training_timer])

        y_pred_prob = model.predict(X_test_fold)
        y_pred = [np.argmax(p_set) for p_set in y_pred_prob]
        n_correct = sum(y_pred == y_test_fold)
        accuracy = n_correct / len(y_pred)
        accuracies.append(accuracy)

        train_time = sum(training_timer.logs)

        print(f"Predictions for fold {counter}:    {np.array(y_pred)}")
        print(f"Actual targets for fold {counter}: {y_test_fold}")
        print(f"Accuracy for fold {counter}: {n_correct}/{len(y_pred)} = {accuracy}")

        print(f"The fold training time is {train_time} seconds")
        counter += 1
        total_correct += n_correct
        total_training_time += train_time

    total_accuracy = total_correct / len(y)
    total_params = model.count_params()
    print(f"Accuracies across all folds:       {accuracies}")
    print(f"Average accuracy across all folds: {np.mean(accuracies)}")
    print(f"Total accuracy across all folds: {total_correct}/{len(y)} = {total_accuracy}")
    print(f"The total training time across all folds: {total_training_time}")
    print(f"The total number of model parameters is: {total_params}")
    return accuracies, total_accuracy, total_training_time, total_params


def classical_cv_score(X, y, clf, name, n_splits=5, scaling=True, loocv=False, debug=False, is_knn=False, is_svm=False, is_dt=False):
    # Initialise colorama.
    init()

    np.random.seed(42)
    if loocv:
        folds = LeaveOneOut()
    else:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True)

    accuracies = []
    counter = 1
    total_correct = 0

    final_predictions = []

    print(f"Performing Classification for {name} Model.")
    for train_index, test_index in folds.split(X, y):

        print(f"\nPerforming ClassicalCV iteration {counter} of {n_splits} for: {name}")

        model = clone(clf)

        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]

        # Apply standardisation.
        if scaling:

            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            # scaler = RobustScaler()
            # scaler = MaxAbsScaler()

            X_train_folds = scaler.fit_transform(X_train_folds)
            X_test_fold = scaler.transform(X_test_fold)

        # Train Model.
        model.fit(X_train_folds, y_train_folds)


        # DEBUG CODE
        if debug:
            import sklearn
            if is_knn:
                        # Determine the used algorithm based on the type of internal data structure

                if isinstance(model._fit_X, sklearn.neighbors._ball_tree.BallTree):
                    used_algorithm = 'ball_tree'
                elif isinstance(model._fit_X, sklearn.neighbors._kd_tree.KDTree):
                    used_algorithm = 'kd_tree'
                else:
                    used_algorithm = 'brute'

                print("Algorithm used:", used_algorithm)

            elif is_svm:
                print(model.get_params())

            elif is_dt:
                print(f"Feature Importances: {model.feature_importances_}")


        # Predict weld classes for testing data.
        y_pred = model.predict(X_test_fold)
        final_predictions.append(y_pred.tolist())

        # Compute the accuracy of predictions.
        n_correct = sum(y_pred == y_test_fold)
        accuracy = n_correct / len(y_pred)
        accuracies.append(accuracy)

        # Hard Voting models cannot produce probabilities.
        try:
            y_pred_prob = model.predict_proba(X_test_fold)
            #print(f"Predicted probabilities for fold {counter}:\n {np.array(y_pred_prob)}")
        except:
            pass

        print(f"Predictions for fold {counter}:    {np.array(y_pred)}")
        print(f"Actual targets for fold {counter}: {y_test_fold}")
        print(f"Accuracy for fold {counter}: {n_correct}/{len(y_pred)} = {accuracy}")
        counter += 1
        total_correct += n_correct

    total_accuracy = total_correct / len(y)

    # Print resutls.
    print(Fore.GREEN)
    print(f"\nResults for the {name} model:")
    print(f"Accuracies across all folds:       {accuracies}")
    print(f"Average accuracy across all folds: {np.mean(accuracies)}")
    print(f"Total accuracy across all folds: {total_correct}/{len(y)} = {total_accuracy}")
    print(Style.RESET_ALL)
    return accuracies, total_accuracy, final_predictions


# Callback to measure the time for each epoch.
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)
