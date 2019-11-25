# %%
# trainとtestどちらかにしか存在しないbuilding_idはない
# abs(中央値-平均値)で分散がよりわかるはず
# 中央値の左の坂の部分のビルは分散の右なのか？
# 二つに割ったあとに時系列をつけてシャッフルしたほうが良い
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

# trainのbuilding_idごとのmeter_readingについて量子化する

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
train=groupon(train, 'building_id', 'meter_reading', 'mean')
train=groupon(train, 'building_id', 'meter_reading', 'median')
train=groupon(train, 'building_id', 'meter_reading', 'var')
display(train)

#%%
print(f'Max:{train["building_id_meter_reading_mean"].unique().max()}')
print(f'Min:{train["building_id_meter_reading_mean"].unique().min()}')
display(plt.hist(train['building_id_meter_reading_mean'].unique(), log=True, bins=200))



# %%
print(f'Max:{train["building_id_meter_reading_median"].unique().max()}')
print(f'Min:{train["building_id_meter_reading_median"].unique().min()}')
display(plt.hist(train['building_id_meter_reading_median'].unique(), log=True, bins=200))

# %%
print(f'Max:{train["building_id_meter_reading_var"].unique().max()}')
print(f'Min:{train["building_id_meter_reading_var"].unique().min()}')
display(plt.hist(train['building_id_meter_reading_var'].unique(), log=True, bins=200))

# %%
func = lambda x: abs(x)
train['abs'] = train['building_id_meter_reading_mean'] - train["building_id_meter_reading_median"]
train['abs'] = train['abs'].map(func)
print(f'Max:{train["abs"].unique().max()}')
print(f'Min:{train["abs"].unique().min()}')
display(plt.hist(train['abs'].unique(), log=True, bins=200))

# %%
merged = train[["building_id", 'building_id_meter_reading_mean', 'building_id_meter_reading_median']]
merged = merged.drop_duplicates().set_index('building_id')
merged.head()

# %%
merged.plot(kind='scatter', x=u'building_id_meter_reading_mean', y=u'building_id_meter_reading_median', log=True)


# %%
