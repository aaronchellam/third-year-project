from pathlib import Path

import pandas as pd
import numpy as np
from graphviz import Source
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from Util.deep_skfold import classical_cv_score
from Util.logger import get_datasets, get_dataset_name, load_stats_data

IMAGE_PATH = Path(__file__).parent.parent / "Graphs"

# datasets = get_datasets()

d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]
feature_names = ["\u03BCP", "\u03C3P", "\u03BCT", "\u03C3T", "\u03BCR", "\u03C3R"]
class_names = ["SW", "LoC", "OP"]


# print(f"Decision Tree CV Scores for Dataset {dname}: ", scores)
# print(f"Decision Tree Average CV Score for Dataset {dname}: ", scores.mean())

# def dt_pca(n_components=2):
#     for idx, dataset in enumerate(datasets):
#
#     scaler = StandardScaler()
#
#
#     pca = PCA(n_components=n_components)



def main():
    for idx, dataset in enumerate(datasets):
        X, y = dataset
        dname = get_dataset_name(idx)

        dt = DecisionTreeClassifier()
        # sk_folds = StratifiedKFold(n_splits=5)
        # scores = classical_cv_score(X, y, dt, "Decision Tree", n_splits=5,
        #                             scaling=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        dt.fit(X_train, y_train)


        out_file = str(IMAGE_PATH / f"dt_{dname}.dot")
        export_graphviz(dt, out_file=out_file, feature_names=feature_names, class_names=class_names,
                        rounded=True, filled=True
                        )

        # Read the content of the DOT file
        source = Source.from_file(out_file)

        # Save the decision tree visualization as a PNG image
        source.render(filename=out_file.replace(".dot", ""), format='png', cleanup=True)


        y_pred = dt.predict(X_test)

        n_correct = sum(y_pred == y_test)
        accuracy = n_correct / len(y_pred)
        print(f"Decision Tree Predictions for {dname}: {y_pred}")
        print(f"Actual Targets: {y_test}")
        print(f"Accuracy: {accuracy}\n")

if __name__ == '__main__':
    main()
