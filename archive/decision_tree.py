import sklearn
from sklearn import tree
import pandas as pd
import numpy as np

# Read data from excel files.
from sklearn.model_selection import train_test_split

xlsx = pd.ExcelFile('../Data/statistic_features.xlsx')
dfa = pd.read_excel(xlsx, 'A')
dfb = pd.read_excel(xlsx, 'B')
dfc = pd.read_excel(xlsx, 'C')

# Extract the feature matrix.
X = dfc.iloc[1:,:6]
y = dfc.iloc[1:,6]

X = X.to_numpy()
y = y.to_numpy()

# Randomise Data Order
i = np.argsort(np.random.random(X.shape[0]))
X = X[i]
y = y[i]

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit Decision Tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
y_predict = dt.predict(X_test)
print(y_predict)
print(y_test)
print(sklearn.metrics.accuracy_score(y_test, y_predict))