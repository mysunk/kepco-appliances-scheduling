import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from util import *
from sklearn.multioutput import MultiOutputRegressor

#%% load dataset
path_home = 'data/'
start = '2012-06-01'
end = '2018-04-01'
data = load_dataset(path_home, start=start, end=end)
# data.to_csv('data_save/train.csv',index=True)

#%% train-test split
x_columns = ['smp', 'brent', 'du', 'lng', 'wti', 'usd', 'gold', 'sup_cap', 'dem_cur',
       'dem_max', 'sup_res_pwr', 'sup_res_ratio', 'ops_res_pwr',
       'ops_res_ratio'] # , 'year', 'month', 'day', 'dayofweek']
y_column = 'smp'

past = 24*28 - 1
future = 24*28

pasts = (np.ones(len(x_columns)) * past).astype(int)
pasts[1:] = 0 # smp만 과거 데이터를 피쳐로

x_col_index = np.zeros(np.shape(x_columns),dtype=int)
for i, x_column in enumerate(x_columns):
    x_col_index[i] = np.where(x_column == data.columns)[0][0]
y_col_index = np.where(y_column == data.columns)[0][0]

#%% train
param = {
    'bagging_freq':10,
    'boosting':'gbdt',
    'colsample_bynode':0.73892,
    'colsample_bytree':0.276,
    'learning_rate':0.03128,
    'max_bin':139,
    'max_depth':-1,
    'metrics':'mape',
    'min_data_in_leaf':124,
    'min_sum_hessian_in_leaf':0.03147,
    'n_jobs':-1,
    'num_leaves':77
         }

# validation
X, y = trans(data.values, pasts, future, x_col_index, y_col_index)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
model = MultiOutputRegressor(lgb.LGBMRegressor(**param), n_jobs=-1)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_val)
# model save (for validation)
save_obj(model, 'model_val.lgb')

#%% train with whole dataset
model = MultiOutputRegressor(lgb.LGBMRegressor(**param), n_jobs=-1)
model.fit(X, y)
# model save (for test)
save_obj(model, 'model_test.lgb')

#%% test
start = '2018-03-04'
end = '2018-04-29'
data = load_dataset(path_home, start=start, end=end)
test, y_true_4 = trans(data.values, pasts, 24*28, x_col_index, y_col_index)
y_pred_4 = model.predict(test)

start = '2018-04-03'
end = '2018-05-29'
data = load_dataset(path_home, start=start, end=end)
test, y_true_5 = trans(data.values, pasts, 24*28, x_col_index, y_col_index)
y_pred_5 = model.predict(test)

result_4 = evaluate(y_true=y_true_4, y_pred = y_pred_4)
result_5 = evaluate(y_true=y_true_5, y_pred = y_pred_5)

#%% plot
start = '2018-03-01'
end = '2018-05-29'
smp = load_smp(path_home, start=start, end=end)

import matplotlib.pyplot as plt
plt.figure()
date_range = pd.date_range('2018-03-01', '2018-04-01', freq='H', closed='left')
plt.plot(date_range,smp.loc[date_range[0]:date_range[-1],:].values)
plt.plot(pd.date_range('2018-04-01', '2018-04-29', freq='H', closed='left'), np.ravel(y_true_4),label='true')
plt.plot(pd.date_range('2018-04-01', '2018-04-29', freq='H', closed='left'), np.ravel(y_pred_4),label='pred')
plt.legend()
plt.xticks(rotation=25)
plt.title('April')
plt.show()

plt.figure()
date_range = pd.date_range('2018-04-01', '2018-05-01', freq='H', closed='left')
plt.plot(date_range,smp.loc[date_range[0]:date_range[-1],:].values)
plt.plot(pd.date_range('2018-05-01', '2018-05-29', freq='H', closed='left'), np.ravel(y_true_5),label='true')
plt.plot(pd.date_range('2018-05-01', '2018-05-29', freq='H', closed='left'), np.ravel(y_pred_5),label='pred')
plt.legend()
plt.xticks(rotation=25)
plt.title('May')
plt.show()