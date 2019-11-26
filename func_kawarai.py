#%%
import numpy as np
import pandas as pd
import itertools
import os, gc
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from IPython import get_ipython
from IPython.core.display import display

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def tuning(train):
    y_rus, X_rus = (train['meter_reading'], train.drop(columns=['meter_reading']))

    # Split the full sample into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.33, random_state=314)


    # early_stopping_rounds にテストサブセットを使用
    # これにより過学習を回避でき、ツリー数の最適化を不必要

    fit_params={"early_stopping_rounds":30, 
                "eval_metric" : 'neg_mean_squared_log_error',
                "eval_set" : [(X_test,y_test)],
                'eval_names': ['valid'],
                #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
                'verbose': 100,
                'categorical_feature': 'auto'}

    # ハイパーパラメータの探索
    # グリッドサーチよりも柔軟で効率的なランダムサーチを採用

    param_test ={
                'learning_rate' : sp_uniform(loc=0.1, scale=0.2),
                'subsample': sp_uniform(loc=0.2, scale=0.6),
                'feature_fraction': sp_uniform(loc=0.6, scale=0.3),
                'min_data_in_leaf':sp_randint(10, 40),
                'max_depth':sp_randint(4, 8),#あとで考える
                }#0.9}

    # This parameter defines the number of HP points to be tested
    # このパラメータは、ハイパーパラメータのテストを行う回数を定義
    iteration = 30

    # n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
    clf = lgb.LGBMRegressor(n_estimators=1000,#6000
                            subsample_freq=1,
                            metric='rmse')

    gs = RandomizedSearchCV(
        estimator=clf, param_distributions=param_test, 
        n_iter=iteration,
        scoring='neg_mean_squared_error',
        cv=4,
        refit=True,
        random_state=314,
        verbose=True)

    gs.fit(X_train, y_train, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
    opt_parameters = gs.best_params_
    
    clf_sw = lgb.LGBMRegressor(**clf.get_params())
    #set optimal parameters
    clf_sw.set_params(**opt_parameters)
    return clf_sw, gs.cv_results_

#def average_imputation(df, column_name):
#    imputation = df.groupby(['timestamp'])[column_name].mean()
#    
#    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)
#    del imputation
#    return df

def add_beaufort_scale(df):
#    df = average_imputation(df, 'wind_speed')
#    df = average_imputation(df, 'wind_direction')

    beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 
            (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

    for item in beaufort:
        df.loc[(df['wind_speed']>=item[1]) & (df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

    df['beaufort_scale'] = df['beaufort_scale'].astype(np.uint8)

    return df
