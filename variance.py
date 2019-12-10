#%%
import gc
import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

%load_ext autoreload
%autoreload 2
from func_ishikawa import *
#%%
path = './submissions'
submission_list = os.listdir(path)
# %% [markdown]
# ![](./img/cov.png)
#%%
comb_list=[i for i in itertools.combinations(submission_list, r=2)]
covariance=[]
for c in comb_list:
    sub1_meter_reading = pd.read_csv(path+'/'+c[0])['meter_reading'].values#ndarray
    sub2_meter_reading = pd.read_csv(path+'/'+c[1])['meter_reading'].values
    cov_result=np.cov(sub1_meter_reading, sub2_meter_reading)[0][1]
    print([int(cov_result), c])
    covariance.append([int(cov_result), c])
    
#%%
print(covariance)