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

import os
import tarfile
import urllib

HOUSING_PATH="./datasets/housing/"

# +
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


# -

housing = load_housing_data()

housing.head()

import matplotlib.pyplot as plt

import numpy as np


def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


from zlib import crc32

from sklearn.model_selection import train_test_split

housing["income_cat"]= pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

from sklearn.model_selection import StratifiedShuffleSplit

from pandas.plotting import scatter_matrix

from pylab import *

#  ### 210121 기록
#  
#  # 2.5 머신러닝 알고리즘을 위한 데이터 준비

# ## 2.5.1 데이터 정제

# 대부분의 머신러닝 알고리즘은 누락된 특성을 다루지 못하므로 이를 처리할 수 있는 함수를 몇개 만들겠습니다. 앞서 total_bedrooms 특성에 값이 없는 경우를 보았는데 이를 고쳐보겠습니다. 방법은 3가지 입니다. 
#
# - 해당 구역을 제거합니다
#
# - 전체 특성을 삭제합니다.
#
# - 어떤 값으로 채웁니다(0,평균,중간값등)
#
# 데이터프레임의 dropna(),drop(),fillna() 메서드를 이용해 이런 작업을 간단하게 처리할 수 있습니다.

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing, housing["income_cat"]):
    start_train_set = housing.loc[train_index]
    start_test_set=housing.loc[test_index]

housing = start_train_set.drop("median_house_value",axis=1)

housing_labels = start_train_set["median_house_value"].copy()

housing

housing.dropna(subset=["total_bedrooms"])

housing.drop("total_bedrooms",axis=1)

median = housing["total_bedrooms"].median()

median

housing["total_bedrooms"].fillna(median,inplace=True)

housing

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

imputer

housing_num = housing.drop("ocean_proximity",axis=1)

housing_num

imputer.fit(housing_num)

imputer.statistics_

housing_num.median().values

X = imputer.transform(housing_num)

housing_tr=pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)

housing_tr

# ## 2.5.2 텍스트와 범주형 특성 다루기 

housing_cat=housing[["ocean_proximity"]]

housing_cat.head(10)

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

housing_cat_encoded[:10]

ordinal_encoder.categories_

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot

housing_cat_1hot.toarray()

cat_encoder.categories_

# ## 2.5.3 나만의 변환기 

# +
from sklearn.base import BaseEstimator, TransformerMixin

room_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6

class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    
