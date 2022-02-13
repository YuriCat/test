
# https://blog.amedama.jp/entry/2018/05/01/081842

import xgboost as xgb

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(y_train)
print(len(y_train))
print(len(y_test))
weights = np.random.randn(len(y_train))

xgb_train = xgb.DMatrix(X_train, y_train) #, weight=weights)
xgb_eval = xgb.DMatrix(X_test, y_test)

xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
}

model = xgb.train(xgb_params, xgb_train)

def eval(model):
    y_pred = model.predict(xgb_eval)
    y_pred_max = np.argmax(y_pred, axis=1)
    accuracy = sum(y_test == y_pred_max) / len(y_test)
    print(accuracy)

eval(model)

xgb_all = xgb.DMatrix(X, y, weight=np.random.random(len(y)))

print(xgb_train)
print(xgb_eval)

xgb_one = xgb.DMatrix(X[:1], y[:1])
xgb_many = xgb.DMatrix(X[:122], y[:122])

model = xgb.Booster(xgb_params, [xgb_many])

for i in range(10):
    X_train, _, y_train, _ = train_test_split(X, y)
    xgb_train = xgb.DMatrix(X_train, y_train, weight=0 * np.random.random(len(y_train)))
    model.update(xgb_train, 0)
    eval(model)

