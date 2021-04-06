import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

train, test = train_test_split(df, test_size=0.25)
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:', len(test))
print('\n')

features = df.columns[:4]

y = pd.factorize(train['species'], sort=True)[0]

clf = RandomForestClassifier(n_jobs=-1, random_state=0)
clf.fit(train[features], y)
predictions = clf.predict(test[features])
print("Prediction probabilities for first 5 examples:")
print(clf.predict_proba(test[features])[:5])
print('\n')

predictions = iris.target_names[predictions]
cm = metrics.confusion_matrix(test['species'], predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="flare")
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.show()

print(list(zip(train[features], clf.feature_importances_)))
