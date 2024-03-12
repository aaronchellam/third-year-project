"""
Create extra data examples via augmentation.
"""
from pathlib import Path

import numpy as np

from Util.deep_logger import load_deep_data
from Util.logger import load_stats_data

DATA_PATH = Path(__file__).parent.parent / "Data"
SAVE_PATH = DATA_PATH / "augmented_statistic_data.npz"
DEEP_SAVE_PATH = DATA_PATH / "deep_augmented_data.npz"

SAVE_PATH_C = DATA_PATH / "C_augmented_statistic_data.npz"
DEEP_SAVE_PATH_C = DATA_PATH / "C_deep_augmented_data.npz"

def augment(X, y):
    # Combine features and labels into a single dataset.
    # data = np.hstack((X, y.reshape(-1, 1)))
    data = [(X[i], y[i]) for i in range(len(y))]

    # Create  dictionary to store the datapoints for each class.
    class_dict = {i: [] for i in range(3)}

    # Populate the class_dict with the datapoints belonging to each class.
    # for row in data:
    #     class_dict[row[-1]].append(row[:-1])

    for features, label in data:
        class_dict[label].append(features)

    # Calculate the average feature vector for each pair of datapoints.
    augmented_X = []
    augmented_y = []

    for class_id, class_data in class_dict.items():
        for i in range(len(class_data)):
            for j in range(i + 1, len(class_data)):
                avg_feature_matrix = (class_data[i] + class_data[j]) / 2
                augmented_X.append(avg_feature_matrix)
                augmented_y.append(class_id)

    augmented_X = np.concatenate((augmented_X, X), axis=0)
    augmented_y = np.concatenate((augmented_y, y), axis=0)

    return augmented_X, augmented_y


def generate_augmented_C(dataset_C, save=False, deep=False):
    X,y = dataset_C

    sw_C = [[X[i], y[i]] for i in range(len(X)) if y[i] == 0]
    sw_C = np.array(sw_C)

    rest_C = [[X[i], y[i]] for i in range(len(X)) if y[i] != 0]
    rest_C = np.array(rest_C)


    augX, augy = augment(list(sw_C[:,0]), list(sw_C[:,1]))

    augmented_X = np.concatenate((list(rest_C[:,0]), augX), axis=0)
    augmented_y = np.concatenate((list(rest_C[:,1]), augy), axis=0)

    if save:
        if deep:
            f = open(DEEP_SAVE_PATH_C, "wb")
        else:
            f = open(SAVE_PATH_C, "wb")

        np.savez(f, augCX=augmented_X, augCy=augmented_y)
        f.close()

def load_augmented_C():
    npzfile = np.load(SAVE_PATH_C)
    augCX = npzfile['augCX']
    augCy = npzfile['augCy']

    return augCX, augCy

def load_deep_augmented_C():
    npzfile = np.load(DEEP_SAVE_PATH_C)
    augCX = npzfile['augCX']
    augCy = npzfile['augCy']

    return augCX, augCy
def generate_augmented_data(datasets, save=False, deep=False):
    augmented_data = {}
    for index, (X, y) in enumerate(datasets):
        augX, augy = augment(X, y)

        savex = f"d{index + 1}X"
        savey = f"d{index + 1}y"

        augmented_data[savex] = augX
        augmented_data[savey] = augy

    if save:
        if deep:
            f = open(DEEP_SAVE_PATH, "wb")
        else:
            f = open(SAVE_PATH, "wb")
        # Use the ** unpacking operator to unpack key-word arguments from dict.
        np.savez(f, **augmented_data)
        f.close()


def load_augmented_data(deep=False):
    if deep:
        npzfile = np.load(DEEP_SAVE_PATH)
    else:
        npzfile = np.load(SAVE_PATH)
    d1X = npzfile['d1X']
    d1y = npzfile['d1y']

    d2X = npzfile['d2X']
    d2y = npzfile['d2y']

    d3X = npzfile['d3X']
    d3y = npzfile['d3y']

    return d1X, d1y, d2X, d2y, d3X, d3y


def main():
    # d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
    d1X, d1y, d2X, d2y, d3X, d3y = load_deep_data()
    datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]
    generate_augmented_C([d2X, d2y], save=True, deep=True)
    augX, augy = load_deep_augmented_C()
    print(augX.shape)
    print(len(augy))
    # generate_augmented_data(datasets, save=True, deep=True)


if __name__ == '__main__':
    main()
