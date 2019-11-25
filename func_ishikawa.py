import numpy as np
import pandas as pd
import itertools
import os, gc
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from IPython import get_ipython
from IPython.core.display import display
import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def load_data(source, path):#testまたはtrainを読み込む
    if not os.path.exists(f'{path}/{source}.pickle'):#pickle
        assert source in ['train', 'test']
        building = pd.read_csv(f'{path}/building_metadata.csv', dtype={'building_id':np.uint16, 'site_id':np.uint8})
        weather  = pd.read_csv( \
                        f'{path}/weather_{source}.csv', parse_dates=['timestamp'], \
                        dtype={'site_id':np.uint8, 'air_temperature':np.float16, 'cloud_coverage':np.float16, 'dew_temperature':np.float16, 'precip_depth_1_hr':np.float16}, \
                    )
        df = pd.read_csv(f'{path}/{source}.csv', dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])
        df = df.merge(building, on='building_id', how='left')#building_metadata.csvと結合
        df = df.merge(weather, on=['site_id', 'timestamp'], how='left')#wether_(train/test).csvと結合
        del building, weather
        gc.collect
        joblib.dump(df, f'{path}/{source}.pickle', compress=True)
    else:
        df=joblib.load(f'{path}/{source}.pickle')
    return df

def groupon(df, group_col, calc_col, method='median'):
    print('{}|{}|{}'.format(group_col, calc_col, method))
    dic=df.groupby(group_col)[calc_col].agg([method])[method].to_dict()
    df[f'{group_col}_{calc_col}_{method}'] = df[group_col].map(dic)
    return df

def allgroupon(df, not_use, category, numerical, method):
    print('ALLgroupon')
    print('not_use:',not_use)
    print('category:',category)
    print('numerical:',numerical)
    print('method:',method)
    for c, n, m in itertools.product(category, numerical, method):
        if not n=='meter_reading':#meter_readingは飛ばす
            print('{}|{}|{}'.format(c, n, m))
            dic=df.groupby(c)[n].agg([m])[m].to_dict()
            df[f'{c}_{n}_{m}']=df[c].map(dic)
    return df

def fillna_numerical(df, not_use, numerical):
    print('fillna')
    for c in numerical:
        if df[c].isnull().any() == True:
            if df[c].dtype == 'float16':
                df[c] = df[c].astype('float32')
            new_num = df[c].mean(skipna=True)
            df[c] = df[c].fillna(new_num)
            print(f'{c}|{new_num}|{df[c].dtype}')
    return df

def transform(df, not_use):
    print('transform')
    #primary_useは施設の利用方法
    df['primary_use'] = np.uint8(LabelEncoder().fit_transform(df['primary_use']))  # encode labels
    # expand datetime into its components
    df['hour'] = np.uint8(df['timestamp'].dt.hour)#dt=dataframeの時系列変換メソッド
    #df['day'] = np.uint8(df['timestamp'].dt.day)
    df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
    # absm=lambda x: abs(x-6)
    # df['month'] = np.uint8(df['timestamp'].dt.month.map(absm))
    #df['month'] = np.uint8(df['timestamp'].dt.month)
    #df['year'] = np.uint8(df['timestamp'].dt.year-2000)

    dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
    df['is_holiday'] = (df['timestamp'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

    # parse and cast columns to a smaller type
    df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)
    df['log_square_feet'] = np.float16(np.log(df['log_square_feet']))#square_feetの対数を取る
    df['year_built'] = np.uint8(df['year_built']-1900)#年数から1900を引く
    df['floor_count'] = np.uint8(df['floor_count'])#uintに
    
    del df['timestamp']

    #ターゲット列を変換
    if 'meter_reading' in df.columns:
        df['meter_reading'] = np.log1p(df['meter_reading']).astype(np.float32) # comp metric uses log errors
    for col in df.columns:
        if col in not_use:
            del df[col]#not_useにある列を削除
    return df

def category_merge(df, cat_A, cat_B):
    under_bar=lambda x: f'{x}_'
    A=df[cat_A].astype(str).map(under_bar)
    B=df[cat_B].astype(str)
    df[f'MERGED_{cat_A}_{cat_B}'] = np.uint8(LabelEncoder().fit_transform(A.str.cat(B)))
    return df

def kill_outlier(df, sigma=2, border_amount=0.0001):
    all_amount=len(df)#データ数
    # 平均と標準偏差
    for c in df.columns:
        if df[c].dtype in ['int8','int16','int32','int64','float16','float32','float64']:
            print('{} is available.'.format(c), end='')
            average = np.mean(df[c].values)
            sd = np.std(df[c].values)

            # 外れ値の基準点
            outlier_min = average - (sd) * sigma
            outlier_max = average + (sd) * sigma

            min_amount=len(df[df[c] < outlier_min])#下方向外れ値の個数
            max_amount=len(df[df[c] > outlier_max])#上方向外れ値の個数
            print('<-{}|{}->'.format(min_amount, max_amount))
            
            print('MIN:{}÷{} < {}'.format(min_amount, all_amount, border_amount))
            print('MAX:{}÷{} < {}'.format(max_amount, all_amount, border_amount))
            if (min_amount / all_amount) < border_amount and min_amount != 0:#min_amountは0ではなく、borderamauntより全データ数に対しての割合が低い
                print('-',end='')
                df = df[df[c] >= outlier_min]#outlier_min以上だけ抽出
            if (max_amount / all_amount) < border_amount and max_amount != 0:#max_amountは0ではなく、borderamauntより全データ数に対しての割合が低い
                print('+', end='')
                df = df[df[c] <= outlier_max]#outlier_max以下だけ抽出
            print(len(df))
        else:
            print('{} is neither integer nor float.'.format(c))
    return df.reset_index(drop=True)


def nakaie_humidity(df):
    func = lambda x: 6.11*(10**((7.5*x)/(237.3+x)))
    df['humidity'] = df['dew_temperature'].map(func)/df['air_temperature'].map(func)
    return df

#%%
def saving(df, holdout_score, importance, heatmap, histogram, comment):
    print('saving')
    time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    out_path = './output/'+time+'/'
    os.mkdir(out_path)
    with open(out_path+str(int(holdout_score*1000))+'.txt', mode='w') as f:
        f.write(comment+'\n')
        f.write(str(df.columns.tolist())+'\n')
        f.write(str(holdout_score)+'\n')
    heatmap.savefig(out_path+'heatmap.png')
    importance.savefig(out_path+'importance.png')
    return out_path

def fit_lgbm(train, val, devices=(-1,), seed=None, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):
    """Train Light GBM model"""
    X_train, y_train = train
    X_valid, y_valid = val
    metric = 'rmse'
    params = {'num_leaves': 31,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': lr,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": bf,
              "feature_fraction":  0.87,
              "metric": metric,
#               "verbosity": -1,
#               'reg_alpha': 0.1,
#               'reg_lambda': 0.3
              }
    device = devices[0]
    if device == -1:
        # use cpu
        pass
    else:
        # use gpu
        print(f'using gpu device_id {device}...')
        params.update({'device': 'gpu', 'gpu_device_id': device})

    params['seed'] = seed

    early_stop = 20
    verbose_eval = 20

    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)
    watchlist = [d_train, d_valid]

    print('training LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)

    # predictions
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    
    print('best_score', model.best_score)
    log = {'train/mae': model.best_score['training']['rmse'],
           'valid/mae': model.best_score['valid_1']['rmse']}
    return model, y_pred_valid, log


def create_X_y(train_df, selected_meter, feature_cols, category_cols):
    train_df = train_df[train_df['meter'] == selected_meter]
    # target_train_df = target_train_df.merge(building_meta_df, on='building_id', how='left')
    # target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
    X_train = train_df[feature_cols + category_cols]
    y_train = train_df['meter_reading_log1p'].values
    del train_df
    return X_train, y_train


def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df
# %%
