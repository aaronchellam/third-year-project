import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from Util.logger import get_datasets, get_dataset_name, load_stats_data

# TODO Format the names of voting classifiers
from Util.util import dict_to_df

classifiers = {0: "DT", 1: "KNN", 2: "SVM", 3: "H-Voting", 4: "S-Voting"}

d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()

datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]


def print_latex(df_k, df_sk, dname, f):
    print(f"Results for K-Fold on {dname}")
    print(df_k.to_markdown(), "\n")
    print(f"Results for Stratified K-Fold on {dname}")
    print(df_sk.to_markdown(), "\n")

    f.write(f"Results for Regular K-Fold on {dname}\n")
    f.write(df_k.to_latex(bold_rows=True))
    f.write("\n")

    f.write(f"Results for Stratified K-Fold on {dname}\n")
    f.write(df_sk.to_latex(bold_rows=True))
    f.write("\n")


def main(latex, debug):
    f = open("tables.tex", "w")
    results = {}
    avg_results = {}

    for idx, dataset in enumerate(datasets):
        X, y = dataset
        dname = get_dataset_name(idx)
        print(f"START OF DATASET {dname}")

        dt = DecisionTreeClassifier()
        knn = KNeighborsClassifier(n_neighbors=3)

        # SVC probability=True makes the SVC class use CV to estimate class probabilities
        svm = SVC(probability=True)

        hard_voting_clf = VotingClassifier(
            estimators=[
                ('dt', dt), ('knn', knn), ('svm', svm)
            ],
            voting='hard'
        )

        soft_voting_clf = VotingClassifier(
            estimators=[
                ('dt', dt), ('knn', knn), ('svm', svm)
            ],
            voting='soft'
        )

        # Create a dictionary to group the results for this dataset.
        kfold_dict = {}
        skfold_dict = {}
        # loo_dict = {}

        for index, clf in enumerate((dt, knn, svm, hard_voting_clf, soft_voting_clf)):
            # Initialise stratified and normal kfold.

            k_folds = KFold(n_splits=10)
            sk_folds = StratifiedKFold(n_splits=10)
            # loo = LeaveOneOut()

            # If clf is a voting clf, change name to differentiate between hard_voting and soft_voting.
            vote_type = ''
            if clf.__class__.__name__ == 'VotingClassifier':
                vote_type = f"({clf.voting})"

            # Implement the scaling pipeline.
            scaler = StandardScaler()
            pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])

            k_fold_scores = cross_val_score(pipeline, X, y, cv=k_folds) * 100
            sk_scores = cross_val_score(pipeline, X, y, cv=sk_folds) * 100
            # loo_scores = cross_val_score(pipeline, X, y, cv=loo) * 100

            kfold_dict[classifiers[index]] = k_fold_scores
            skfold_dict[classifiers[index]] = sk_scores
            # loo_dict[classifiers[index]] = np.mean(loo_scores)

            if debug:
                print("\n")
                print(f"{clf.__class__.__name__} {vote_type} Standard CV Scores for Dataset {dname}: ", k_fold_scores)
                print(f"{clf.__class__.__name__} Average Standard CV Score for Dataset {dname}:", k_fold_scores.mean())

                print("\n")
                print(f"{clf.__class__.__name__} {vote_type} Stratified CV Scores for Dataset {dname}: ", sk_scores)
                print(f"{clf.__class__.__name__} Average Stratified CV Score for Dataset {dname}:", sk_scores.mean())

                # print("\n")
                # print(f"{clf.__class__.__name__} {vote_type} LOOCV Scores for Dataset {dname}: ", loo_scores)
                # print(f"{clf.__class__.__name__} Average LOOCV Score for Dataset {dname}:", loo_scores.mean())

        # Process kfold and skfold dictionaries into dataframes.
        df_k = dict_to_df(kfold_dict)
        df_sk = dict_to_df(skfold_dict)
        # df_loo = pd.DataFrame(loo_dict)

        if latex:
            print_latex(df_k, df_sk, dname, f)

        results[dname] = [df_k, df_sk]
        avg_results[dname] = [df_k["AVG"].to_dict(), df_sk["AVG"].to_dict()]


    df = pd.DataFrame(avg_results, index=["kfold", "skfold"])

    generate_barplots(df)

    f.close()


def generate_barplots(df):
    # Kfold barplot.
    ax = pd.DataFrame(df.loc['kfold'].to_dict()).plot.bar(figsize=(11, 6), rot=0)
    for container in ax.containers:
        ax.bar_label(container)
    plt.ylim([0, 100])
    plt.title("K-Fold Accuracy vs. Classifier", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Classifier")
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('../Graphs/kfold.png')
    plt.show()

    # Skfold barplot.
    ax = pd.DataFrame((df.loc['skfold'].to_dict())).plot.bar(figsize=(11, 6), rot=0)
    for container in ax.containers:
        ax.bar_label(container)
    plt.ylim([0, 100])
    plt.title("Stratified K-Fold Accuracy vs. Classifier")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Classifier")
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('../Graphs/skfold.png')
    plt.show()


if __name__ == '__main__':
    latex = False
    debug = True

    main(latex, debug)
