from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from Util.deep_skfold import classical_cv_score
from Util.logger import load_stats_data

d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]



def main():
    for idx, dataset in enumerate(datasets):
        X, y = dataset



        a1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=10000, learning_rate=0.1)
        a2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=500, learning_rate=0.1)
        a3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=1)
        names = ["a1"]
        rfs = [a1]

        for idx2, rf in enumerate(rfs):
            accuracies, total_accuracy, final_predictions = classical_cv_score(X, y, rf, names[idx2], n_splits=5,
                                                                               scaling=True)




if __name__ == '__main__':
    main()


