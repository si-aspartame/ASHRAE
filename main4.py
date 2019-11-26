# %%
#メーターごとに学習して、バリデーションは1:1
import gc
import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


%load_ext autoreload
%autoreload 2
from func_ishikawa import *
#結果は1.41
#testデータ2年分を四分割し、春夏用と秋冬用のモデルを分けたもの
#↑がmain2だが、今回は分割したのちに重みを付けてブレンドする

#%%
path = './input'
train = load_data('train', path)
category=['building_id','primary_use', 'meter','site_id', "hour", "weekday"]
not_use=['row_id','index', False, 'wind_direction', 'wind_speed', 'sea_level_pressure']
numerical=[x for x in train.columns if x not in category+not_use]
print(f' category_cols:{category}\n numerical_cols:{numerical}')

#%%
train = transform(train, not_use)
train = reduce_mem_usage(train, use_float16=True)

#%%
y_train = train['meter_reading']
X_train = train.drop(['meter_reading'], axis=1)
print(f'columns:{train.columns}')

#%%
X_half_1 = X_train[:int(X_train.shape[0] / 2)]#split features in half
X_half_2 = X_train[int(X_train.shape[0] / 2):]

y_half_1 = y_train[:int(X_train.shape[0] / 2)]#split labels in half
y_half_2 = y_train[int(X_train.shape[0] / 2):]

#%%
categorical_features = ["building_id", "site_id", "meter", "primary_use", "hour", "weekday"]
d_half_1 = lgb.Dataset(X_half_1, label=y_half_1, categorical_feature=categorical_features, free_raw_data=False)
d_half_2 = lgb.Dataset(X_half_2, label=y_half_2, categorical_feature=categorical_features, free_raw_data=False)

#%%
#lgb.trainのvalid_setsには学習に使ったデータも重複して引数にする必要がある
watchlist_1 = [d_half_1, d_half_2]
watchlist_2 = [d_half_2, d_half_1]

#%%
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 40,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse"
}
print("FIRST...")
model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=watchlist_1, verbose_eval=200, early_stopping_rounds=200)
print("SECOND...")
model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=watchlist_2, verbose_eval=200, early_stopping_rounds=200)
print("DONE!")

#%%
# del X_train, y_train, X_half_1, X_half_2, y_half_1, y_half_2, d_half_1, d_half_2, watchlist_1, watchlist_2
# gc.collect()

#%%[markdown]
## 以下はテスト

#%%
test = load_data('test', path)
row_ids = test['row_id']
test = transform(test, not_use)
test = reduce_mem_usage(test, use_float16=True)
print(f'columns:{test.columns}')

#%%
one, two, three, four = [int(test.shape[0] / 4), int(test.shape[0] / 4)*2, int(test.shape[0] / 4)*3, int(test.shape[0] / 4)*4]
print(f'{one}, {two}, {three}, {four}')

#%%
test_split_1 = test[:one]
test_split_2 = test[one:two]
test_split_3 = test[two:three]
test_split_4 = test[three:]
#%%
pred_one = (np.expm1(model_half_1.predict(test_split_1, num_iteration=model_half_1.best_iteration)) \
    *0.6) \
    + \
    (np.expm1(model_half_2.predict(test_split_1, num_iteration=model_half_2.best_iteration)) \
    * 0.4)
pred_two = (np.expm1(model_half_2.predict(test_split_2, num_iteration=model_half_2.best_iteration)) \
    *0.6) \
    + \
    (np.expm1(model_half_1.predict(test_split_2, num_iteration=model_half_1.best_iteration)) \
    * 0.4)
pred_three = (np.expm1(model_half_1.predict(test_split_3, num_iteration=model_half_1.best_iteration)) \
    *0.6) \
    + \
    (np.expm1(model_half_2.predict(test_split_3, num_iteration=model_half_2.best_iteration)) \
    * 0.4)
pred_four = (np.expm1(model_half_2.predict(test_split_4, num_iteration=model_half_2.best_iteration)) \
    *0.6) \
    + \
    (np.expm1(model_half_1.predict(test_split_4, num_iteration=model_half_1.best_iteration)) \
    * 0.4)
    
#%%
pred = np.concatenate([pred_one, pred_two, pred_three, pred_four])

#%%
if not test.shape[0] == len(pred):
    print(f'T{test.shape[0]}:P{len(pred)}')
else:
    print('length is OK')

#%%
submission = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(pred, 0, a_max=None)})
submission.to_csv(f"{os.path.basename(__file__)}.csv", index=False)
print('DONE!')


# %%
