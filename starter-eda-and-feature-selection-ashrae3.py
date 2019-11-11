# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
# I'll summarise here the insights from my brief data analyses and whatever is shared in public discussions.
# This is still highly under progress and will be updated when possible.
# 
# Quoting Chris Balbach (our Competition Host):
# > Consider the scenario laid out in this diagram. This competition simulates the modelling challenge presented at the end of the timeline when measured energy and weather conditions are known and the adjusted baseline energy must be calculated. 
# ![Energy Consumption/Demand vs Time](https://www.mdpi.com/make/make-01-00056/article_deploy/html/images/make-01-00056-g001-550.jpg)
# > For further details, I recommend reading [this paper](https://www.mdpi.com/2504-4990/1/3/56) by Clayton Miller.
# 
# We are actually using data from 2016 to predict the demand for both 2017 and 2018, which is a hard task. However, other public kernels have showed that you can give a relatively good estimate even with a very simple linear model.

#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, gc
import random
import datetime

from tqdm import tqdm_notebook as tqdm

# matplotlib and seaborn for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.offline as offline
# offline.init_notebook_mode()

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.externals import joblib

import lightgbm as lgb


#%%
path = './input'
# Input data files are available in the "../input/" directory.
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#%% [markdown]
# # Load data and display samples

#%%
unimportant_cols = ['wind_direction', 'wind_speed', 'sea_level_pressure']
target = 'meter_reading'
def load_data(source='train', path=path):#testまたはtrainを読み込む
    if not os.path.exists(f'{path}/{source}.pickle'):#pickle
        assert source in ['train', 'test']
        building = pd.read_csv(f'{path}/building_metadata.csv', dtype={'building_id':np.uint16, 'site_id':np.uint8})
        weather  = pd.read_csv( \
                        f'{path}/weather_{source}.csv', parse_dates=['timestamp'], \
                        dtype={'site_id':np.uint8, 'air_temperature':np.float16, 'cloud_coverage':np.float16, 'dew_temperature':np.float16, 'precip_depth_1_hr':np.float16}, \
                        usecols=lambda c: c not in unimportant_cols
                    )
        df = pd.read_csv(f'{path}/{source}.csv', dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])
        df = df.merge(building, on='building_id', how='left')#building_metadata.csvと結合
        df = df.merge(weather, on=['site_id', 'timestamp'], how='left')#wether_(train/test).csvと結合
        joblib.dump(df, f'{path}/{source}.pickle', compress=True)
    else:
        df=joblib.load(f'{path}/{source}.pickle')
    return df
train = load_data('train')
train.sample(7)

#%% [markdown]
# ### Test data

#%%
test = load_data('test')
test.sample(7)

#%% [markdown]
# # EDA and sample statistics

#%%
print(f'Training from {train.timestamp.min()} to {train.timestamp.max()}, and predicting from {test.timestamp.min()} to {test.timestamp.max()}')


#%%
# target's log-log histogram:
ax = np.log1p(train.meter_reading).hist()#自然対数、真数に+1、多分1未満だとバグるから？
ax.set_yscale('log')

# describe raw values first
train.meter_reading.describe()


#%%
# check the distribution in the types of meters
meters = train.groupby('building_id').meter.nunique()
plt.title('Distribution of types of meters\n{0:electricity, 1:water, 2:steam, 3:hotwater}') # from the official starter kernel
_ = meters.hist()
# from the graphs it looks like steam and hotwater are reversed (e.g.: 3:steam, 2:hotwater) but that shouldn't make any difference to the model

#%% [markdown]
# ### display a single time series (notice measurement errors and discontinuities)

#%%
#例として4つのメーターがあるID1258をピックアップする
building_id = 1258  # a building with all 4 meters
#4つのメーターがある行のメータ値のユニークな値の数＝電気水道蒸気温水の4つ
meters = train[train['building_id'] == building_id].meter.nunique()
print('meters:',meters)

#%%
#同じビル、４つのメーター、違う日のデータがたくさんある
display(train[train['building_id'] == building_id])

#%%
#点が気温、線がメータ値、0電気1水道2蒸気3温水
for meter in range(meters):
    fig, ax = plt.subplots()
    plt.title(f'Building {building_id} Meter {meter}')
    ax2 = ax.twinx()
    # plot meter_reading
    idx = (train['building_id'] == building_id) & (train['meter'] == meter)
    dates = matplotlib.dates.date2num(train.loc[idx, 'timestamp'])
    ax2.plot_date(dates, train.loc[idx, 'meter_reading'], '-', label='meter_reading', alpha=0.8)
    # plot air_temperature
    dates = matplotlib.dates.date2num(train.loc[train['building_id'] == building_id, 'timestamp'])
    ax.plot_date(dates, train.loc[train['building_id'] == building_id, 'air_temperature'], '.', color='tab:cyan', label='air_temperature')
    ax.set_ylabel('air_temperature'); ax2.set_ylabel('meter_reading')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')

#%% [markdown]
# ### now let's see what's the expected prediction in the test set for the same building
# ### 右下の黄色い棒のところが予測しなければいけないテストデータ
#%%
for meter in range(meters):
    train_sample = train[(train['building_id'] == building_id) & (train['meter'] == meter)]  # same train sample as above

    test['meter_reading'] = 0.0
    test_sample = test[(test['building_id'] == building_id) & (test['meter'] == meter)]  # and the same meter in the test set

    fig, ax = plt.subplots(figsize=(16,4))
    plt.title(f'Meter {meter}')
    ax.xaxis.set_tick_params(rotation=30, labelsize=10)
    ax2 = ax.twinx()

    # plot training sample
    dates = matplotlib.dates.date2num(train_sample['timestamp'])
    ax2.plot_date(dates, train_sample['meter_reading'], '-', label='train', alpha=0.8)
    ax.plot_date(dates, train_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')

    # plot test sample
    dates = matplotlib.dates.date2num(test_sample['timestamp'])
    ax2.plot_date(dates, test_sample['meter_reading'], '*', label='test', alpha=0.8)
    ax.plot_date(dates, test_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')

    ax.set_ylabel('air_temperature'); ax2.set_ylabel('meter_reading')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    del train_sample; del test_sample; del dates


#%%
# some feature stats
train.describe()


#%%
# the counts above expose the missing data (Should we drop or refill the missing data?)
print("trainのNANじゃない割合:")
data_ratios = train.count()/len(train)#count()はデータの個数をカウント、nanはカウントされない
data_ratios


#%%
# Is the same happening in the test set? Yes
print("testのNANじゃない割合:")
test.count()/len(test)


#%%
# we can refill with averages
train.loc[:, data_ratios < 1.0].mean()#NaNを含む列の値の平均

#%% [markdown]
# # Preprocess data

#%%
class ASHRAE3Preprocessor(object):
    @classmethod
    def fit(cls, df, data_ratios=data_ratios):#nanがあるものの平均とラベルエンコーダを定義
        cls.avgs = df.loc[:,data_ratios < 1.0].mean()
        cls.pu_le = LabelEncoder()
        cls.pu_le.fit(df["primary_use"])

    @classmethod
    def transform(cls, df):
        df = df.fillna(cls.avgs) # refill NAN with averages
        #primary_useは施設の利用方法
        df['primary_use'] = np.uint8(cls.pu_le.transform(df['primary_use']))  # encode labels

        # expand datetime into its components
        df['hour'] = np.uint8(df['timestamp'].dt.hour)#dt=dataframeの時系列変換メソッド
        df['day'] = np.uint8(df['timestamp'].dt.day)
        df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
        df['month'] = np.uint8(df['timestamp'].dt.month)
        df['year'] = np.uint8(df['timestamp'].dt.year-2000)
        
        # parse and cast columns to a smaller type
        df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)
        df['log_square_feet'] = np.float16(np.log(df['log_square_feet']))#square_feetの対数を取る
        df['year_built'] = np.uint8(df['year_built']-1900)#年数から1900を引く
        df['floor_count'] = np.uint8(df['floor_count'])#uintに
        
        # remove redundant columns
        for col in df.columns:
            if col in ['timestamp', 'row_id']:
                del df[col]#timestampとrow_idを削除
    
        # extract target column
        #ターゲット列を変換
        if 'meter_reading' in df.columns:
            df['meter_reading'] = np.log1p(df['meter_reading']).astype(np.float32) # comp metric uses log errors
        return df

#%%        
ASHRAE3Preprocessor.fit(train)
train = ASHRAE3Preprocessor.transform(train)
train.sample(7)


#%%
train.dtypes

#%% [markdown]
# # Feature ranked correlation

#%%
fig, ax = plt.subplots(figsize=(16,8))
# use a ranked correlation to catch nonlinearities
corr = train[[col for col in train.columns if col != 'year']].sample(100100).corr(method='spearman')
_ = sns.heatmap(corr, annot=True, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

#%% [markdown]
# # Train K folds

#%%
# force the model to use the weather data instead of dates, to avoid overfitting to the past history
# 指定した4つの列以外の列を学習に使う（落とさない）
features = [col for col in train.columns if col not in [target, 'year', 'month', 'day']]

##########################################以下は学習
#%%
folds = 4
seed = 42
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
# oof_pred = np.zeros(train.shape[0])  # out of fold predictions
models = []

## stratify data by building_id
for tr_idx, val_idx in tqdm(kf.split(train, train['building_id']), total=folds):
    def fit_regressor(tr_idx, val_idx):
        tr_x, tr_y = train[features].iloc[tr_idx], train[target][tr_idx]#有効な列の値と答え
        vl_x, vl_y = train[features].iloc[val_idx], train[target][val_idx]#上のバリデーション
        print({'train size':len(tr_x), 'eval size':len(vl_x)})
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)
        clf = lgb.LGBMRegressor(n_estimators=6000,
                                learning_rate=0.28,
                                feature_fraction=0.9,
                                subsample=0.2,  # batches of 20% of the data
                                subsample_freq=1,
                                num_leaves=20,
                                metric='rmse')
        clf.fit(tr_x, tr_y,
                eval_set=[(vl_x, vl_y)],
                early_stopping_rounds=50,
                verbose=200)
#         oof_pred[val_idx] += clf.predict(vl_x, num_iteration=clf.best_iteration_)
        return clf
    clf = fit_regressor(tr_idx, val_idx)
    models.append(clf)
#     break ### TODO: remove to run next folds
    
gc.collect()

#%% [markdown]
# # Feature importance

#%%
_ = lgb.plot_importance(models[0], importance_type='gain')


#%%
# calculate the mean feature importance, so that we can update 'unimportant_cols' above
feature_importance = np.mean([m._Booster.feature_importance(importance_type='gain') for m in models], axis=0)
sorted(zip(feature_importance, train.columns), reverse=True)

#%% [markdown]
# # Check prediction

#%%
# load and pre-process test data
test = ASHRAE3Preprocessor.transform(test)
test.sample(7)

#%% [markdown]
# ### now let's revisit the same meter we had initially looked at, and check the predictions for each fold

#%%
def recover_timestamp(x):
    ''' reassemble timestamp using date components '''
    return datetime.datetime.strptime(f'{x.year}-{x.month}-{x.day} {x.hour}', '%y-%m-%d %H')

fig, ax = plt.subplots(figsize=(16,4))
plt.title(f'Building {building_id} Meter {meter} on all {folds} prediction folds')
ax.xaxis.set_tick_params(rotation=30, labelsize=10)
ax2 = ax.twinx()

train_sample = train[(train['building_id'] == building_id) & (train['meter'] == meter)]  # same training sample as before
test_sample = test[(test['building_id'] == building_id) & (test['meter'] == meter)]   # and the same meter in the test set

# plot training sample
dates = matplotlib.dates.date2num(train_sample[['year', 'month', 'day', 'hour']].apply(recover_timestamp, axis=1))
ax.plot_date(dates, train_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')
ax2.plot_date(dates, np.expm1(train_sample['meter_reading']), '-', color='tab:blue', label='train')

# plot prediction sample
dates = matplotlib.dates.date2num(test_sample[['year', 'month', 'day', 'hour']].apply(recover_timestamp, axis=1))
ax.plot_date(dates, test_sample['air_temperature'], '.', color='tab:cyan', label='air_temperature')
for i,model in enumerate(models):
    ax2.plot_date(dates, np.expm1(model.predict(test_sample[features])), '-', label=f'prediction{i}', alpha=0.4)

ax.set_ylabel('air_temperature'); ax2.set_ylabel('meter_reading (+prediction)')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')
_ = plt.show()

del test_sample; del train_sample
_ = gc.collect()


#%%
# # Check if all test buildings and meters are the same as in the training data
# train_buildings = np.unique(train[['building_id', 'meter']].values, axis=0)
# # del train; gc.collect()

# test_buildings  = np.unique(test[['building_id', 'meter']].values, axis=0)

# print(len(train_buildings), len(test_buildings))
# [b for b in test_buildings if b not in train_buildings]

#%% [markdown]
# # Test Inference and Submission

#%%
# split test data into batches
set_size = len(test)
iterations = 50
batch_size = set_size // iterations

print(set_size, iterations, batch_size)
assert set_size == iterations * batch_size


#%%
meter_reading = []
for i in tqdm(range(iterations)):
    pos = i*batch_size
    fold_preds = [np.expm1(model.predict(test[features].iloc[pos : pos+batch_size])) for model in models]
    meter_reading.extend(np.mean(fold_preds, axis=0))

print(len(meter_reading))
assert len(meter_reading) == set_size

#%% [markdown]
# ### Save submission

#%%
submission = pd.read_csv(f'{path}/sample_submission.csv')
submission['meter_reading'] = np.clip(meter_reading, a_min=0, a_max=None) # clip min at zero


#%%
submission.to_csv('submission.csv', index=False)
submission.head(9)


#%%
# prediction's log-log histogram:
plt.yscale('log')
_ = np.log1p(submission['meter_reading']).hist()

# describe raw submission
submission['meter_reading'].describe()

