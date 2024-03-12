import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from Util.logger import get_datasets, get_dataset_name
#TODO:The least populated class in y has only 9 members, which is less than n_splits=10.

datasets = get_datasets()


def perform_classification(clf):
    results = []
    for idx, dataset in enumerate(datasets):
        X, y = dataset

        sk_folds = StratifiedKFold(n_splits=10)
        sk_scores = cross_val_score(clf, X, y, cv=sk_folds)
        sk_scores_mean = np.mean(sk_scores)

        results.append((sk_scores, sk_scores_mean))

    return results
