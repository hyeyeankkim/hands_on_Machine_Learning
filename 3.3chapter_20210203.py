# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 3. 분류

# # 3.1 MNIST

import numpy as np


# +
def sort_by_target(mnist): 
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1] 
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1] 
    mnist.data[:60000] = mnist.data[reorder_train] 
    mnist.target[:60000] = mnist.target[reorder_train] 
    mnist.data[60000:] = mnist.data[reorder_test + 60000] 
    mnist.target[60000:] = mnist.target[reorder_test + 60000] 

try: 
    from sklearn.datasets import fetch_openml 
    mnist = fetch_openml('mnist_784', version=1, cache=True) 
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings 
    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset 
except ImportError: 
    from sklearn.datasets import fetch_mldata 
    mnist = fetch_mldata('MNIST original')


# -

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)

mnist.keys()

X,y=mnist["data"],mnist["target"]

X.shape

y.shape

y = y.astype(np.int8)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit =X[0]
some_digit_image=some_digit.reshape(28,28)

plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")
plt.show()

y[0]

X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

X_train

y_train

# # 3.2 이진 분류기 훈련

y_train_5 = (y_train == 5)

y_train_5

y_test_5 = (y_test == 5)

y_test_5

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

from sklearn.linear_model import SGDClassifier 
sgd_clf = SGDClassifier(max_iter=5, random_state=42) 
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

# 에러 : https://somjang.tistory.com/entry/%ED%95%B8%EC%A6%88%EC%98%A8%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-127%ED%8E%98%EC%9D%B4%EC%A7%80-MNIST-%EC%BD%94%EB%93%9C-ValueError-The-number-of-classes-has-to-be-greater-than-one-got-1-class-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95

# # 3.3 성능 측정

# ## 3.3.1 교차 검증을 사용한 정확도 측정

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# +
skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle= True)

for train_index, test_index in skfolds.split(X_train,y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
# -

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy")

# +
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self,X,y=None):
        return self
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)


# -

never_5_clf = Never5Classifier()

cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring="accuracy")

# ## 3.3.2 오차 행렬

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=5)

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5,y_train_pred)

y_train_perfect_predictions = y_train_5

confusion_matrix(y_train_5,y_train_perfect_predictions)

# ## 3.3.3 정밀도와 재현율

# +
from sklearn.metrics import precision_score,recall_score

precision_score(y_train_5,y_train_pred)
# -

recall_score(y_train_5,y_train_pred)

from sklearn.metrics import f1_score

f1_score(y_train_5,y_train_pred)

# ## 정밀도/재현율 트레이드오프

y_scores = sgd_clf.decision_function([some_digit])

y_scores

threshold =0

y_some_digit_pred


