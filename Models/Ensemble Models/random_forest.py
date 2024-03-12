from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from Util.deep_skfold import classical_cv_score
from Util.logger import load_stats_data

d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]


def main():
    for idx, dataset in enumerate(datasets):
        X, y = dataset



        rf1 = RandomForestClassifier(n_estimators=100, max_samples=0.9)
        rf2 = RandomForestClassifier(n_estimators=100)
        rf3 = RandomForestClassifier(n_estimators=500, max_samples=0.9)
        rf4 = ExtraTreesClassifier(n_estimators=100)

        names = ["rf1", "rf2", "rf3"]
        rfs = [rf1, rf2, rf3]
        for idx2, rf in enumerate(rfs):
            accuracies, total_accuracy, final_predictions = classical_cv_score(X, y, rf, names[idx2], n_splits=5,
                                                                               scaling=True, is_dt=True, debug=True)





if __name__ == '__main__':
    main()