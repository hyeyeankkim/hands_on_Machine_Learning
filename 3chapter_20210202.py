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


