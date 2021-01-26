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


train_set,test_set = split_train_test(housing,0.2)

from zlib import crc32


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier))& 0xfffffff < test_ratio * 2**32


def split_train_test_by_id(data,test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]


housing_with_id = housing.reset_index()
train_set,test_set = split_train_test_by_id(housing_with_id, 0.2,"index")

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2,random_state=42)

housing["income_cat"]= pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

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
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
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

from sklearn.compose import ColumnTransformer

# +
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs),
])
# -

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared

# # 2.6 모델 선택과 훈련

# ## 2.6.1 훈련 세트에서 훈련하고 평가하기 

# +
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
# -

some_data = housing.iloc[:5]

some_data

some_labels= housing_labels.iloc[:5]

some_labels

some_data_prepared = full_pipeline.transform(some_data)

print("예측 : ", lin_reg.predict(some_data_prepared))

print("레이블 : ", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels,housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse

# 대부분 구역의 중간 주택 가격은 120,000에서 265,000사이입니다. 그러므로 예측 오차가 68,628인 것은 매우 만족스럽지 못합니다. 이는 모델이 훈련 데이터에 과소적합된 사례입니다. 이런 상황은 특성들이 좋은 예측을 만들 만큼 충분한 정보를 제공하지 못했거나 모델이 충분히 강력하지 못하다는 사실을 말해줍니다. 

# +
from sklearn.tree import DecisionTreeRegressor

tree_reg= DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
# -

housing_predictions= tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels,housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse

# ## 2.6.2 교차 검증을 사용한 평가

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)

tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("점수 : " , scores)
    print("평균 : " , scores.mean())
    print("표준편차 : ", scores.std())


display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_scores = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)

# ** tip 
#
# 실험한 모델을 모두 저장해두면 필요할 때 쉽게 모델을 복원할 수 있습니다. 교차 검증 점수와 실제 예측값은 물론 하이퍼파라미터와 훈련된 모델 파라미터 모두 저장해야 합니ㅏㄷ. 이렇게 하면 여러 모델의 점수와 모델이 만든 오차를 쉽게 비교할 수 있습니다. 파이썬의 pickle 패키지나 큰 넘파이 배열을 저장하는 데 아주 효율적인 joplib(pip를 사용해 이 라이브러리를 설치할 수 있습니다)을 사용해서 사이킷런 모델을 간단하게 저장할 수 잇습니다.

import joblib

joblib.dump(my_model,"my_model.pkl")
#나중에 불러오기 
my_model_load = joblib.load("my_model.pkl")

# # 2.7 모델 세부 튜닝

# ## 2.7.1 그리드 탐색

from sklearn.model_selection import GridSearchCV

param_grid =[
    {'n_estimators':[3,10,30],'max_features' : [2,4,6,8]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg,param_grid, cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_

for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)

# ## 2.7.2 랜덤 탐색
#
# 그리드 탐색 방법은 이전 예제와 같이 비교적 적은 수의 조합을 탐구할 때 괜찮습니다. 하지만, 하이퍼파라미터 탐색 공간이 커지면 RandomizedSearchCV를 사용하는 편이 더 좋습니다. RandomizedSearchCV는 Grid_SearchCV와 거의 같은 방식으로 사용하지만 가능한 모든 조합을 시도하는 대신 각 반복마다 하이퍼파라미터에 임의의 수를 대입하여 지정한 횟수만큼 평가합니다. 
#
# - 랜덤 탐색을 1,000회 반복하도록 실행하면 하이퍼파라미터마다 각기 다른 1,000개의 값을 탐색합니다.(그리드 탐색에서는 하이퍼파라미터마다 몇 개의 값만 탐색합니다).
#
# - 단순히 반복 횟수를 조절하는 것만으로도 하이퍼 파라미터 탐색에 투입할 컴퓨팅 자원을 제어할 수 있습니다. 
#
# ## 2.7.3 앙상블 방법 
#
# 모델을 세밀하게 튜닝하는 또 다른 방법은 최상의 모델을 연결해보는 것입니다. 모델의 그룹이 최상의 단일 모델보다 더 나은 성능을 발휘할 때가 많습니다.

# ## 2.7.4 최상의 모델과 오차 분석

feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances

extra_attribs=["rooms_per_hhold","pop_per_hhold","bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances,attributes),reverse=True)

# ## 2.7.5 테스트 세트로 시스템 평가하기

final_model = grid_search.best_estimator_

final_model

X_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_predictions

final_mse = mean_squared_error(y_test,final_predictions)


