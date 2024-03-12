import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC, SVC

from Util.deep_skfold import classical_cv_score
from Util.logger import get_dataset_name, get_datasets, load_stats_data

# TODO: use CV to pick best C hyper-parameter for SVM.
# TODO: experiment with different loss functions.
#TODO: Source for the plots.




d1X, d1y, d2X, d2y, d3X, d3y = load_stats_data()
# d1X, d1y, d2X, d2y, d3X, d3y = load_augmented_data()
datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]

def main():
    for idx, dataset in enumerate(datasets):
        X, y = dataset
        dname = get_dataset_name(idx)

        svm = SVC(verbose=True)
        # sk_folds = StratifiedKFold(n_splits=5)
        # scores = cross_val_score(svm, X, y, cv=sk_folds)
        #
        # print(f"Linear SVM CV Scores for Dataset {dname}: ", scores)
        # print(f"Linear SVM Average CV Score for Dataset {dname}:", scores.mean())

        classical_cv_score(X, y, svm, 'SVM', n_splits=5,
                           scaling=True, debug=True, is_svm=True)


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_

    #svs = np.array( [[x,y] for x,y in svs if x < 4])

    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


def linear_svm_plot():

    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    # SVM Classifier model
    svm_clf = SVC(kernel="linear", C=1000)
    svm_clf.fit(X, y)

    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bo", label="Class 1")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Class 2")
    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)
    plt.title("Linear SVC", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.axis([0, 5.8, 0, 2])


    plt.show()


def svm_outlier_plot():
    np.random.seed(42)
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    y = iris["target"]

    X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
    y_outliers = np.array([0, 0])
    Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
    yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
    Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
    yo2 = np.concatenate([y, y_outliers[1:]], axis=0)

    svm_clf2 = SVC(kernel="linear", C=10**9)
    svm_clf2.fit(Xo2, yo2)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)

    plt.sca(axes[0])
    plt.plot(Xo1[:, 0][yo1==1], Xo1[:, 1][yo1==1], "bo")
    plt.plot(Xo1[:, 0][yo1==0], Xo1[:, 1][yo1==0], "yo")
    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)
    plt.annotate("Outlier",
                 xy=(X_outliers[0][0], X_outliers[0][1]),
                 xytext=(2.5, 1.7),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=16,
                 )
    plt.axis([0, 5.5, 0, 2])

    plt.sca(axes[1])
    plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "bo")
    plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "yo")
    plot_svc_decision_boundary(svm_clf2, 0, 5.5)
    plt.xlabel("Feature 1", fontsize=14)
    plt.annotate("Outlier",
                 xy=(X_outliers[1][0], X_outliers[1][1]),
                 xytext=(3.2, 0.08),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=16,
                 )
    plt.axis([0, 5.5, 0, 2])
    plt.tight_layout()
    plt.show()


from mpl_toolkits.mplot3d import Axes3D

def plot_3D_decision_function(ax, w, b, x1_lim=[4, 6], x2_lim=[0.8, 2.8], X=[], y=[]):
    x1_in_bounds = (X[:, 0] > x1_lim[0]) & (X[:, 0] < x1_lim[1])
    X_crop = X[x1_in_bounds]
    y_crop = y[x1_in_bounds]
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    xs = np.c_[x1.ravel(), x2.ravel()]
    df = (xs.dot(w) + b).reshape(x1.shape)
    m = 1 / np.linalg.norm(w)
    boundary_x2s = -x1s*(w[0]/w[1])-b/w[1]
    margin_x2s_1 = -x1s*(w[0]/w[1])-(b-1)/w[1]
    margin_x2s_2 = -x1s*(w[0]/w[1])-(b+1)/w[1]
    ax.plot_surface(x1s, x2, np.zeros_like(x1),
                    color="r", alpha=0.2, cstride=100, rstride=100)

    ax.plot_surface(x1s, x2, np.zeros_like(x1)+1,
                    color="b", alpha=0.2, cstride=100, rstride=100)

    ax.plot_surface(x1s, x2, np.zeros_like(x1)-1,
                    color="y", alpha=0.2, cstride=100, rstride=100)

    ax.plot(x1s, boundary_x2s, 0, "k-", linewidth=2, label=r"$h=0$")
    ax.plot(x1s, margin_x2s_1, 0, "k--", linewidth=2, label=r"$h=\pm 1$")
    ax.plot(x1s, margin_x2s_2, 0, "k--", linewidth=2)
    ax.plot(X_crop[:, 0][y_crop==1], X_crop[:, 1][y_crop==1], 0, "bo")
    ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
    ax.plot(X_crop[:, 0][y_crop==0], X_crop[:, 1][y_crop==0], 0, "yo")
    ax.axis(x1_lim + x2_lim)
    ax.text(4.5, 2.5, 4.2, "Decision function $h$", fontsize=16)
    ax.set_xlabel(r"Feature 1", fontsize=16, labelpad=10)
    ax.set_ylabel(r"Feature 2", fontsize=16, labelpad=10)
    ax.set_zlabel(r"$h = \mathbf{w}^T \mathbf{x} + b$", fontsize=18, labelpad=5)
    ax.legend(loc="upper left", fontsize=16)

def make_3D_plot():
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris virginica


    svm_clf2 = SVC(kernel="linear", C=2)
    svm_clf2.fit(X, y.ravel())
    print(svm_clf2.intercept_, svm_clf2.coef_)
    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    plot_3D_decision_function(ax1, w=svm_clf2.coef_[0], b=svm_clf2.intercept_[0], X=X, y=y)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    make_3D_plot()