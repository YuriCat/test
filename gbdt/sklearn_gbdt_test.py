
# https://blog.amedama.jp/entry/2018/05/01/081842

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(y_train)
print(len(y_train))
weights = np.random.randn(len(y_train))

model = GradientBoostingClassifier()
model = model.fit(X_train, y_train)

def eval(model):
    y_pred = model.predict_proba(X_test)
    y_pred_max = np.argmax(y_pred, axis=1)
    accuracy = sum(y_test == y_pred_max) / len(y_test)
    print(accuracy)

eval(model)
