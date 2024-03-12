import numpy as np
import itertools
from colorama import init, Fore, Style

from sklearn.base import ClassifierMixin, clone
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.utils import all_estimators, compute_class_weight

# Estimators
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from tabulate import tabulate

from Util.augmentation import load_augmented_data, load_augmented_C
from Util.deep_skfold import classical_cv_score
from Util.logger import load_stats_data
from Util.util import dict_to_df

# TODO: If gaussian process is used in hard-voting, we can use one-vs-one.

# TODO: Return to non-augmented data
d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]

dCX, dCy = load_augmented_C()
datasets = [[dCX, dCy]]
# d1X, d1y, d2X, d2y, d3X, d3y = load_augmented_data(deep=False)
# datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]


class_weights = [compute_class_weight("balanced", classes=np.unique(d[1]), y=d[1]) for d in datasets]
priors = [np.bincount(d[1]) / sum(np.bincount(d[1])) for d in datasets]

names = [
    "Decision Tree",
    "KNN",
    "RBF SVM",
    "Gaussian Process",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "Gradient Boosting",
]


# TODO: Hyperparameter tuner?

def print_latex(df_sk, d_index, f):
    f.write(f"Results for D{d_index + 1}\n")
    f.write(df_sk.to_latex(bold_rows=True))
    f.write("\n")


def main(latex=True):
    #f = open("tables2.tex", "w")

    # Iterate through each of the three datasets.
    for idx1, dataset in enumerate(datasets):
        print(f"\nSTART OF DATASET {str(idx1 + 1)}\n")

        # Extract features and labels.
        X = dataset[0]
        y = dataset[1]

        # cw = class_weights[idx1]

        # Model priors change depending on class balance.
        p = priors[idx1]

        # Classifier list with hyperparameters set.
        classifiers = [
            DecisionTreeClassifier(),
            KNeighborsClassifier(n_neighbors=3),
            SVC(probability=True),
            GaussianProcessClassifier(1.0 * RBF(1.0), multi_class="one_vs_rest"),
            RandomForestClassifier(n_estimators=100, max_samples=0.9),
            AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=2), n_estimators=100,
                learning_rate=0.1),
            GaussianNB(priors=p),
            QuadraticDiscriminantAnalysis(priors=p),
            GradientBoostingClassifier()
        ]

        # Store the accuracies and the predicted classes in dictionaries.
        skfold_dict = {}
        results_dict = {}
        predictions_dict = {}

        # Compute performance of all classifiers.
        for idx2, (name, clf) in enumerate(zip(names, classifiers)):
            # scaler = StandardScaler()
            # scaler = MinMaxScaler()
            # scaler = RobustScaler()
            # scaler = MaxAbsScaler()
            # pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])

            # Accuracy from cross-validation.
            accuracies, total_accuracy, final_predictions = classical_cv_score(X, y, clf, name, n_splits=5,
                                                                               scaling=True)
            # Store the accuracies and the predicted classes
            skfold_dict[name] = accuracies
            results_dict[name] = total_accuracy
            predictions_dict[name] = final_predictions

            # print(f"\nAccuracies of Classifier {name}: {accuracies}")
            # print(f"Average Accuracy: {np.mean(accuracies)}\n")

        df_sk = dict_to_df(skfold_dict)
        results = list(results_dict.items())
        headers = ['Classification Model', 'Total Accuracy']
        table = tabulate(results, headers=headers, tablefmt='fancy_grid')

        print(f"\nClassification Results for Dataset {str(idx1 + 1)}:")
        print(table)
        print(tabulate(results, headers=headers, tablefmt='latex'))

        #if latex:
            #print_latex(df_sk, idx1, f)

        # Print predictions for all models on this dataset.
        # for name, predictions in predictions_dict.items():
        #     print(f"Predictions for {name}:")
        #     print(predictions, "\n")

        # Find the best voting classifier for dataset 1.
        # if idx1 == 0:
        best_ensemble(X, y, classifiers)

    #f.close()


def best_ensemble(X, y, classifiers, n_splits=5, loocv=False):
    init()
    # Compute every 3-combination of classifiers.
    clf_combinations = list(itertools.combinations(classifiers, 3))
    clf_combination_names = list(itertools.combinations(names, 3))

    voting_results = {}
    total_combinations = len(clf_combinations)
    # Compute the performance of each voting ensemble.
    for idx, ensemble in enumerate(clf_combinations):
        # Extract the three models + names.
        clf1 = clone(ensemble[0])
        clf2 = clone(ensemble[1])
        clf3 = clone(ensemble[2])

        name1 = clf_combination_names[idx][0]
        name2 = clf_combination_names[idx][1]
        name3 = clf_combination_names[idx][2]

        # Construct voting models.
        hv = VotingClassifier(
            estimators=[
                (name1, clf1), (name2, clf2), (name3, clf3)
            ],
            voting='hard'
        )
        hv_name = f"HV({name1} | {name2} | {name3})"

        sv = VotingClassifier(
            estimators=[
                (name1, clf1), (name2, clf2), (name3, clf3)
            ],
            voting='soft'
        )
        sv_name = f"SV({name1} | {name2} | {name3})"

        # Compute accuracy of voting models.
        print(Fore.YELLOW)
        print(f"Performing Classification for Ensemble {idx + 1}/{total_combinations}")
        print(Style.RESET_ALL)
        hv_accuracies, hv_total_accuracy, hv_final_predictions = classical_cv_score(X, y, hv, hv_name,
                                                                                    n_splits=n_splits,
                                                                                    scaling=True, loocv=loocv)

        sv_accuracies, sv_total_accuracy, sv_final_predictions = classical_cv_score(X, y, sv, sv_name,
                                                                                    n_splits=n_splits,
                                                                                    scaling=True, loocv=loocv)

        # Record results.
        voting_results[hv_name] = hv_total_accuracy
        voting_results[sv_name] = sv_total_accuracy

    # Sort the voting results in descending order of value.
    voting_results = dict(sorted(voting_results.items(), key=lambda item: item[1], reverse=True))

    # Print the best 10 voting models.
    top_10 = list(voting_results.items())[:10]
    headers = ['Voting Model', 'Total Accuracy']
    table = tabulate(top_10, headers=headers, tablefmt='fancy_grid')

    print("\nTop 10 Voting Models:")
    print(table)
    print(tabulate(top_10, headers=headers, tablefmt='latex'))
    return


if __name__ == '__main__':
    p = priors[0]
    classifiers = [
        DecisionTreeClassifier(),
        KNeighborsClassifier(n_neighbors=3),
        SVC(probability=True),
        GaussianProcessClassifier(1.0 * RBF(1.0), multi_class="one_vs_rest"),
        RandomForestClassifier(n_estimators=100, max_samples=0.9),
        AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=2), n_estimators=100,
            learning_rate=0.1),
        GaussianNB(priors=p),
        QuadraticDiscriminantAnalysis(priors=p),
        GradientBoostingClassifier()

    ]
    # best_ensemble(d1X, d1y, classifiers, loocv=True)
    main()
