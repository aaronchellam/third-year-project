from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from Util.generic_classifier import perform_classification
import pandas as pd

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=0.9, bootstrap=True)

results = perform_classification(bag_clf)
scores = []
means = []

for score, mean in results:
    scores.append(score * 100)
    means.append(mean * 100)
    print(score)
    print(mean)

df = pd.DataFrame(scores)
df["Average"] = means

df = df.round(1)
df.index += 1

print(df.to_markdown())

ax = df["Average"].plot.bar(rot=0)
for container in ax.containers:
    ax.bar_label(container)
plt.ylim([0, 100])
plt.title("Decision Tree Bagging Classifier")
plt.ylabel("Accuracy (%)")
plt.xlabel("Dataset")
# plt.legend(bbox_to_anchor=(1, 0.5))
plt.tight_layout()
# plt.savefig('../Graphs/dtbag.png')
plt.show()

f = open("bagging.tex", "w")
f.write(df.to_latex())
f.close()
