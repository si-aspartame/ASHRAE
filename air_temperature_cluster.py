# %%
import gc
import os
import random
import jupyter

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

#%%
from func_ishikawa import load_data, groupon
%load_ext autoreload
%autoreload 2
%matplotlib inline

# %%
path = './input'
train = load_data('train', path)

# %%
category = ['building_id', 'primary_use', 'meter', 'site_id', "hour", "weekday"]
not_use = ['row_id', 'index', False, 'wind_direction', 'wind_speed', 'sea_level_pressure']
numerical = [x for x in train.columns if x not in category+not_use]
print(f' category_cols:{category}\n numerical_cols:{numerical}')

# %%
test = load_data('test', path)

# %%
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
train[train['meter'] == 0]['air_temperature'].plot(ax=axes[0, 0])
train[train['meter'] == 1]['air_temperature'].plot(ax=axes[1, 0])
train[train['meter'] == 2]['air_temperature'].plot(ax=axes[0, 1])
train[train['meter'] == 3]['air_temperature'].plot(ax=axes[1, 1])

# %%
train = groupon(train, 'site_id', 'air_temperature', method='mean')
train = groupon(train, 'site_id', 'air_temperature', method='median')
merged = train[["site_id", 'building_id_air_temperature_mean', 'building_id_air_temperature_median']]
merged = merged.drop_duplicates()

#%%
sns.pairplot(x_vars=['building_id_air_temperature_mean'], y_vars=['building_id_air_temperature_median'], data=merged, hue="site_id", size=5)

