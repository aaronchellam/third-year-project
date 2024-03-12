from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# TODO: Does data need to be scaled.
# TODO: Datasets A and B contain duplicates!!!!!
# TODO: Write a test to ensure that AUBUC has the correct labels.
# TODO: Make sure all datasets have the correct number of samples
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent.parent / "Data"  # os.path.dirname(__file__)  os.path.abspath('../Data')
STATS_PATH = DATA_PATH / 'statistic_features.xlsx'
SAVE_PATH = DATA_PATH / "statistic_data.npz"

dataset_map = {0: 'D1', 1: 'D2', 2: 'D3'}
label_to_int = {'SW': 0, 'LoC': 1, 'OP': 2}
int_to_label = {v: k for k, v in label_to_int.items()}


def parse_data():
    xlsx = pd.ExcelFile('../Data/statistic_features.xlsx')
    d1 = pd.read_excel(xlsx, 'A')
    dfb = pd.read_excel(xlsx, 'B')
    d2 = pd.read_excel(xlsx, 'C')
    d3 = pd.read_excel(xlsx, 'AUBUC')

    return [d1, d2, d3]


# TODO: verify code unaffected by using integers instead of labels
def get_datasets():
    datasets = parse_data()
    parsed_datasets = []

    for idx, dataset in enumerate(datasets):
        # Remove the first line which contains Nans
        dataset = dataset.tail(-1)

        # Dataset 3 needs to be parsed differently.
        if idx == 2:
            dfa = dataset.iloc[:46, :]
            dfb = dataset.iloc[46:56, :]
            dfc = dataset.iloc[56:, :]

            dfa = dfa.sort_values(by=['Signal ID'])
            dfb = dfb.sort_values(by=['Signal ID'])
            dfc = dfc.sort_values(by=['Signal ID'])
            dataset = pd.concat([dfa, dfb, dfc], ignore_index=True)

        # Parse Dataset 1 and 2 normally.
        else:
            dataset = dataset.sort_values(by=['Signal ID'])

        X = dataset.iloc[:, :6]
        y = dataset.loc[:, 'Label']

        y = np.array([label_to_int[i] for i in y])

        X = X.to_numpy()
        # y = y.to_numpy()

        parsed_datasets.append((X, y))

    return parsed_datasets


def sort_dictionary(d):
    sorted_dict = {k: d[k] for k in sorted(d)}
    return sorted_dict


def get_dataset_name(index):
    return dataset_map[index]


def df_map(x):
    return ' '.join(x)


def print_datasets_table():
    parsed_datasets = get_datasets()
    grand_dict = defaultdict(list)
    dict_list = []
    total_datapoints = []

    for (X, y) in parsed_datasets:
        new_y = [label_to_int[label] for label in y]
        label_counts = dict(Counter(new_y))
        dict_list.append(label_counts)
        total_datapoints.append(len(y))
        print(label_counts)

    for idx, d in enumerate(dict_list):
        for key, value in d.items():
            grand_dict[key].append([str(value), f"({round((value / total_datapoints[idx]) * 100)}%)"])

    grand_dict = sort_dictionary(grand_dict)

    df = pd.DataFrame(grand_dict)
    df = df.applymap(df_map)
    columns = [('Class Proportions', 'SW'), ('Class Proportions', 'LoC'), ('Class Proportions', 'OP')]
    df.columns = pd.MultiIndex.from_tuples(columns)
    print(df.to_markdown())

    f = open('Latex/data_overview.tex', 'w')
    f.write(df.to_latex())
    f.close()

    # for i in range(1,4):
    #     new_y.count(i)


# print_datasets_table()

# df = pd.read_csv('../Data/dataset_C/666.csv', skiprows=9, sep=';')
# xlsx = pd.ExcelFile('../Data/statistic_features.xlsx')
# dfa = pd.read_excel(xlsx, 'A')
# dfb = pd.read_excel(xlsx, 'B')
# dfc = pd.read_excel(xlsx, 'C')
#
# # Retrieve row containing Signal id 666
# id_666 = dfc[dfc['Signal ID'] == 666]
#
# # Print the label for id 666
# label = id_666['Label'][1]


def main(save=False):
    data = get_datasets()
    d1x = data[0][0]
    d1y = data[0][1]

    d2x = data[1][0]
    d2y = data[1][1]

    d3x = data[2][0]
    d3y = data[2][1]

    if save:
        # Must open file in write and binary mode.
        f = open(SAVE_PATH, "wb")
        np.savez(f, d1X=d1x, d1y=d1y, d2X=d2x, d2y=d2y, d3X=d3x, d3y=d3y)
        f.close()


def load_stats_data():
    npzfile = np.load(SAVE_PATH)
    d1X = npzfile['d1X']
    d1y = npzfile['d1y']

    d2X = npzfile['d2X']
    d2y = npzfile['d2y']

    d3X = npzfile['d3X']
    d3y = npzfile['d3y']

    return d1X, d1y, d2X, d2y, d3X, d3y


# Compare the stats logger to the deep logger.
def test_loggers():
    from Util.deep_logger import load_deep_data

    d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
    D1X, D1y, D2X, D2y, D3X, D3y = load_deep_data()

    print(d1y == D1y)
    print(d2y == D2y)
    print(d3y == D3y)


# Create 10 sets of 80:20 train-test sets
def create_train_test_sets():
    d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
    stats_data = [(d1X, d1y), (d2X, d2y), (d3X, d3y)]
    for X, y in stats_data:
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


        print('\n')
    print("\n")


if __name__ == '__main__':
    #create_train_test_sets()
    # main()
    # test_loggers()
    pass
