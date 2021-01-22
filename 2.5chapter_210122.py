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

# 조합 특성을 추가하는 간단한 변환기 
#
# 변환기가 add_bedrooms_per_room 하이퍼파라미터 하나를 가지며 기본값을 True로 지정합니다. 합리적인 기본값을 주는 것이 좋습니다. 
# 데이터 준비 단계를 자동화할수록 더 많은 조합을 자동으로 시도해볼 수 있고 최상의 조합을 찾을 가능성을 매우 높여줍니다. 그리고 시간도 많이 절약됩니다.

# +
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True): #*args나 **kargs가 아닙니다. 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None) : 
        return self #더 할 일이 없습니다.
    def transform(self,X):
        rooms_per_household = X[:,rooms_ix] / X[:, households_ix]
        population_per_household = X[:,population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributeAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# -

housing_extra_attribs

# ## 2.5.4 특성 스케일링(feature scaling) 

# 모든 특성의 범위를 같도록 만들어주는 방법으로 min-max 스케일링과 표준화(standardization)가 널리 사용됩니다.
#
# min-max 스케일링이 가장 간단합니다. 많은 사람들이 이를 정규화(normalization)라고 부릅니다. 0~1 범위에 들도록 값을 이동하고 스케일을 조정하면 됩니다. 데이터에서 최솟값을 뺀 후 최댓값과 최솟값의 차이로 나누면 이렇게 할 수 있습니다. 
#
# 표준화는 먼져 평균을 뺀 후(그래서 표준화를 하면 항상 평균이 0이 됩니다) 표준편차로 나누어 결과 분포의 분산이 1이 되도록 합니다. 표준화는 범위의 상한과 하한이 없어 어떤 알고리즘에서는 문제가 될 수 있습니다.(예를 들어 신경망은 종종 입력값의 범위로 0에서 1사이를 기대합니다.)그러나 표준화는 이상치에 영향을 덜 받습니다. 
#
# **주의 : 모든 변환기에서 스케일링은 (테스트세트가 포함된) 전체 데이터가 아니고 훈련 데이터에 대해서만 fit() 메서드를 적용해야 합니다. 그런 다음 훈련 세트와 테스트 세트(그리고 새로운 데이터)에 대해 transform() 메서드를 사용합니다. 

# ## 2.5.5 변환 파이프라인

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler',StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

housing_num_tr


