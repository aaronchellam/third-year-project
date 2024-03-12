import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid

from Util.augmentation import load_augmented_data
from Util.deep_skfold import classical_cv_score
from Util.logger import get_datasets, get_dataset_name, load_stats_data

d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
d1X, d1y, d2X, d2y, d3X, d3y = load_augmented_data()
datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]



def print_algorithm():
    for idx, dataset in enumerate(datasets):
        X, y = dataset

        knn = KNeighborsClassifier(n_neighbors=3)

        knn_weighted = KNeighborsClassifier(n_neighbors=3, weights='distance')
        nc = NearestCentroid()
        classical_cv_score(X, y, knn, 'K-Nearest Neighbours', n_splits=5,
                           scaling=True, debug=True, is_knn=True)








def main():
    for idx, dataset in enumerate(datasets):
        X, y = dataset
        dname = get_dataset_name(idx)

        knn = KNeighborsClassifier(n_neighbors=3)

        sk_folds = StratifiedKFold(n_splits=5)
        scores = cross_val_score(knn, X, y, cv=sk_folds)

        print(f"3NN CV Scores for Dataset {dname}: ", scores)
        print(f"3NN Average CV Score for Dataset {dname}:", scores.mean())


if __name__ == '__main__':
    print_algorithm()