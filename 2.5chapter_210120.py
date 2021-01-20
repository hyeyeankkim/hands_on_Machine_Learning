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

 housing.info()

housing["ocean_proximity"].value_counts()

housing.describe()

# %matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50,figsize=(20,15))
plt.show()

import numpy as np


def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set,test_set = split_train_test(housing,0.2)

len(train_set)

len(test_set)

from zlib import crc32


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier))& 0xfffffff < test_ratio * 2**32


def split_train_test_by_id(data,test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]


housing_with_id = housing.reset_index()
train_set,test_set = split_train_test_by_id(housing_with_id, 0.2,"index")

housing_with_id

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2,random_state=42)

housing["income_cat"]= pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

strat_test_set["income_cat"].value_counts()/len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)

housnig = strat_train_set.copy()

housing.plot(kind="scatter",x="longitude",y="latitude")

housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)

housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
            s=housing["population"]/100, label="population",figsize=(10,7),
            c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,sharex=False)
plt.legend()

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes=["median_house_value","median_income","total_rooms","housing_median_age"]

from pylab import *

scatter_matrix(housing[attributes],figsize(12,8))

housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix=housing.corr()

corr_matrix

corr_matrix["median_house_value"].sort_values(ascending=False)


