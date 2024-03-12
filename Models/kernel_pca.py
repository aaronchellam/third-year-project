import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from Util.logger import get_datasets, get_dataset_name

datasets = get_datasets()

for idx, dataset in enumerate(datasets):
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    kernel_pca = KernelPCA(n_components=60, kernel="rbf", fit_inverse_transform=True)
    # kernel_pca.fit(X_train)
    # X_train_transformed = kernel_pca.transform(X_train)
    # X_test_transformed = kernel_pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)

    pipeline = Pipeline([('transformer', kernel_pca), ('estimator', knn)])
    sk_folds = StratifiedKFold(n_splits=10)
    sk_scores = cross_val_score(pipeline, X, y, cv=sk_folds)

    print(sk_scores)
    print(np.mean(sk_scores), "\n")

    sk_scores = cross_val_score(knn, X, y, cv=sk_folds)
    print(sk_scores)
    print(np.mean(sk_scores), "\n")

