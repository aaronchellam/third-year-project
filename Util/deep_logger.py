import numpy as np
import pandas as pd
import os
from pathlib import Path

# TODO: Use assert to debug this. Debug every step aggressively since some files may have different formats.
# TODO: Assert length of final data matrix.
# TODO: Check shapes - some of dataset A have 20001 and others have 15001
# TODO: Once data has been properly tested, save it so that it can be swiftly retrieved.
# TODO: Debug labels.
# TODO: Note that the deep logger orders the data set by id number which is different from the normal logger - this needs to be addressed if we want to compare the deep models with the classical models.
# TODO: Datasets A and B contain duplicates!!!!!
# TODO: Write a test to ensure that AUBUC has the correct labels.
from sklearn.model_selection import train_test_split

from Util.logger import label_to_int

DATA_PATH = Path(__file__).parent.parent / "Data" #os.path.dirname(__file__)  os.path.abspath('../Data')
STATS_PATH = DATA_PATH / 'statistic_features.xlsx'
SAVE_PATH = DATA_PATH / "saved_data.npz"

filemap = {0: "deep_d1.npz", 1: "deep_d2.npz", 2: "deep_d3.npz"}

def parse_dataset_A():
    """
    This is a test function used to debug the deep_parse() function.
    :return:
    """
    # Path to data examples.
    directory = DATA_PATH / 'dataset_A'

    # Collect the filenames for all data examples.
    filenames = os.listdir(directory)
    data_matrices = []

    assert len(filenames) == 46

    # Extract and format the data from each data example.
    for filename in filenames:
        # Extract.
        data_path = directory / filename #os.path.join(directory, filename)
        data = pd.read_csv(data_path, skiprows=9, sep=';')

        # assert list(data.columns) == ['Index', 'Time', 'P Raw', 'P Eps', 'T Raw', 'T Eps', 'BR Raw', 'BR Eps', 'AN Power']
        assert data.Index[0] == 0

        # Select the six-tuple containing the signal features. (Columns)
        # Select the indices corresponding to the signal window. (Rows)
        # For dataset A, there should be 2500 time-indexed values in total.
        data_matrix = data.iloc[7083:9583, [2, 4, 6]].to_numpy()
        assert data_matrix.shape == (2500, 3)
        data_matrices.append(data_matrix)

    return data_matrices


def deep_parse(letter, num_examples):
    directory = DATA_PATH / f'dataset_{letter}'
    # directory = os.path.join(DATA_PATH, f'dataset_{letter}')
    filenames = os.listdir(directory)

    # The range of indices to splice over depends on the dataset.
    if letter in ['A', 'B']:
        index_start = 7083
        index_end = 9583



    elif letter == 'C':
        index_start = 5000
        index_end = 7500

    else:
        raise Exception("Dataset letter must be one of: 'A', 'B' or 'C'.")

    # Data X.
    data_matrices = []

    # Target label y.
    label_names = []

    # Open the statistics file to map each signal id to its corresponding label.
    xlsx = pd.ExcelFile(STATS_PATH)

    # Use tail(-1) to drop the first row which contains Nan values (excel spreadsheet used merged cells for header).
    statistics = pd.read_excel(xlsx, letter).tail(-1)

    # For each file name, append the data and the label.
    for filename in filenames:

        # Retrieve label.
        signal_id = int(os.path.splitext(filename)[0])

        # Ignore the duplicates in dataset B.
        if letter == 'B' and int(signal_id) in [340, 341, 343, 344]:
            continue

        assert statistics[statistics['Signal ID'] == signal_id].shape[0] == 1
        signal_label = statistics[statistics['Signal ID'] == signal_id].Label.iloc[0]
        label_names.append(signal_label)

        # Retrieve data.
        data_path = directory / filename
        # data_path = os.path.join(directory, filename)
        data = pd.read_csv(data_path, skiprows=9, sep=';')

        # TODO: This test case does not pass due to differing column headers.
        # assert list(data.columns) == ['Index', 'Time', 'P Raw', 'P Eps', 'T Raw', 'T Eps', 'BR Raw', 'BR Eps', 'AN Power']
        assert data.Index[0] == 0

        data_matrix = data.iloc[index_start:index_end, [2, 4, 6]].to_numpy()
        assert data_matrix.shape == (2500, 3)
        data_matrices.append(data_matrix)

    # Convert labels to integers before returning.
    label_names = [label_to_int[i] for i in label_names]

    assert len(label_names) == num_examples

    return np.array(data_matrices), np.array(label_names)


def get_deep_datasets(debug=False, save=False):
    # Parse datasets A and C.
    d1X, d1y = deep_parse('A', 46)
    d2X, d2y = deep_parse('C', 86)

    # Create dataset 3 = AUBUC.
    dbX, dby = deep_parse('B', 10)
    d3X = np.concatenate((d1X, dbX, d2X))
    #d3X = d1X + dbX + d2X
    d3y = np.concatenate((d1y, dby, d2y))
    #d3y = d1y + dby + d2y

    # Perform unit test of parse function.
    if debug:
        d1_test = parse_dataset_A()
        assert np.array_equal(d1_test, d1X)
        assert len(d3X) == 142

        # Test labels are correct.
        # TODO Finish this test.
        xlsx = pd.ExcelFile(STATS_PATH)
        statistics = pd.read_excel(xlsx, 'AUBUC').tail(-1)

    if save:
        # Must open file in write and binary mode.
        f = open(SAVE_PATH, "wb")
        np.savez(f, d1X=d1X, d1y=d1y, d2X=d2X, d2y=d2y, d3X=d3X, d3y=d3y)
        f.close()

    return [[d1X, d1y], [d2X, d2y], [d3X, d3y]]

def load_deep_data():
    npzfile = np.load(SAVE_PATH)
    d1X = npzfile['d1X']
    d1y = npzfile['d1y']

    d2X = npzfile['d2X']
    d2y = npzfile['d2y']

    d3X = npzfile['d3X']
    d3y = npzfile['d3y']

    return d1X, d1y, d2X, d2y, d3X, d3y

def create_deep_train_sets():
    d1X, d1y, d2X, d2y, d3X, d3y = load_deep_data()
    deep_data = [(d1X, d1y), (d2X, d2y), (d3X, d3y)]
    for idx, (X, y) in enumerate(deep_data):
        X_trains = []
        X_tests = []
        y_trains = []
        y_tests = []
        # 10 sets of train-test data required for each dataset
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_trains.append(X_train)
            X_tests.append(X_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

        X_trains = np.array(X_trains)
        X_tests = np.array(X_tests)
        y_trains = np.array(y_trains)
        y_tests = np.array(y_tests)
        save_file = DATA_PATH / filemap[idx]
        f = open(save_file, "wb")
        np.savez(f, X_trains=X_trains, X_tests=X_tests, y_trains=y_trains, y_tests=y_tests)
        f.close()

def load_deep_train_sets():
    npz1 = np.load(DATA_PATH / "deep_d1.npz")
    npz2 = np.load(DATA_PATH / "deep_d2.npz")
    npz3 = np.load(DATA_PATH / "deep_d3.npz")

    data = []

    for i in range(3):
        savefile = DATA_PATH / filemap[i]
        npzfile = np.load(savefile)
        X_trains = npzfile['X_trains']
        X_tests = npzfile['X_tests']
        y_trains = npzfile['y_trains']
        y_tests = npzfile['y_tests']

        data.append((X_trains, X_tests, y_trains, y_tests))

    return data




# TODO check if this needs to be removed later.
if __name__ == '__main__':
    deep_parse('C', 86)


