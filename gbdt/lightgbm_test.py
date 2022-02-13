
# https://blog.amedama.jp/entry/2018/05/01/081842

import lightgbm as lgb

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(y_train)
print(len(y_train))
weights = np.random.randn(len(y_train))

lgb_train = lgb.Dataset(X_train, y_train) #, weight=weights)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgbm_params = {
    'objective': 'multiclass',
    'num_class': 3,
}

model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

def eval(model):
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_max = np.argmax(y_pred, axis=1)
    accuracy = sum(y_test == y_pred_max) / len(y_test)
    print(accuracy)

eval(model)

lgb_all = lgb.Dataset(X, y, weight=np.random.randn(len(y)))
model = lgb.Booster(lgbm_params, lgb_all)

for _ in range(10):
    X_train, _, y_train, _ = train_test_split(X, y)
    lgb_train = lgb.Dataset(X_train, y_train, weight=np.random.randn(len(y_train)))
    model.update(lgb_train)
    print(model.num_trees())
    eval(model)

for _ in range(10):
    X_train, _, y_train, _ = train_test_split(X, y)
    model.refit(X_train, y_train, weight=np.random.randn(len(y_train)))
    print(model.num_trees())
    eval(model)
