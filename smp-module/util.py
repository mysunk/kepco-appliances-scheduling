import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
try:
    import cPickle as pickle
except BaseException:
    import pickle
import os
import matplotlib.pyplot as plt

def load_smp(path, start, end):
    date_range = pd.date_range(start, end, freq='D', closed='left')
    smp_raw = pd.read_csv(path + 'SMP.csv')
    smp_raw = smp_raw.loc[::-1, :].reset_index(drop=True)  # flip
    smp_raw['Time'] = pd.to_datetime(smp_raw['Time'])
    smp_raw.index = smp_raw['Time']
    smp = pd.DataFrame(columns=smp_raw.columns)
    smp['Time'] = date_range
    smp.index = date_range
    smp.loc[date_range, '1':'mean'] = smp_raw.loc[date_range, '1':'mean'].values
    smp.fillna(method='pad', inplace=True)
    smp_d = smp.loc[:, '1':'24'].values
    smp_mean = smp['mean'].values # 가중평균 피쳐로 넣어보기
    # daily -> hourly
    smp_h = pd.DataFrame(columns=['smp'])
    date_range = pd.date_range(start, end, freq='H', closed='left')
    smp_h['smp'] = np.ravel(smp_d)
    smp_h.index = date_range
    return smp_h

def load_pwr_sup_dem(path, start, end):
    path =  path + '전력수급/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    power = []
    for i, file in enumerate(files):
        tmp = pd.read_csv(path + file, encoding='unicode_escape')
        tmp.columns = ['time', 'sup_cap', 'dem_cur', 'dem_max', 'sup_res_pwr', 'sup_res_ratio', 'ops_res_pwr',
                       'ops_res_ratio']
        power.append(tmp)
    power = pd.concat(power, axis=0, ignore_index=True)
    power.index = pd.to_datetime(power['time'], format='%Y%m%d%H%M%S')
    date_range = pd.date_range(start, end, freq='H', closed='left')
    power = power.loc[date_range[0]:date_range[-1],'sup_cap':'ops_res_ratio']
    date_range = pd.date_range(start, end, freq='5min', closed='left')
    power_5min = pd.DataFrame(columns=power.columns)
    power_5min['time'] = date_range
    power_5min.index = date_range
    power_5min.loc[power.index, :] = power
    del power_5min['time']
    power_5min.fillna(method='pad', inplace=True)  # 직전값으로 padding
    # 5분단위 -> 1시간 단위
    power_h = pd.DataFrame(columns=power_5min.columns)
    for col in power_5min.columns:
        power_h[col] = power_5min[col].values.reshape(-1, 12).mean(axis=1)
    date_range = pd.date_range(start, end, freq='H', closed='left')
    power_h.index = date_range
    return power_h

def load_oil(path, start, end):
    path = path + '유가/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    date_range = pd.date_range(start, end, freq='D', closed='left')
    oil = pd.DataFrame(columns=['brent', 'du', 'lng', 'wti', 'usd', 'gold'])
    oil['time'] = date_range
    oil.index = date_range
    del oil['time']
    for file in files:
        tmp = pd.read_csv(path + file)
        tmp['date'] = pd.to_datetime(tmp['date'])
        date_drop = np.where(tmp['date'] > pd.to_datetime(date_range[-1]))[0]
        tmp.drop(index=date_drop, inplace=True)
        tmp = tmp.reset_index(drop=True)
        date_drop = np.where(tmp['date'] < pd.to_datetime(start))[0]
        tmp.drop(index=date_drop, inplace=True)
        tmp = tmp.reset_index(drop=True)
        oil_name = file[:-4]
        oil.loc[tmp['date'], oil_name] = tmp[oil_name].values
    oil.fillna(method='pad', inplace=True)  # 직전값으로 padding
    oil.fillna(method='bfill', inplace=True)  # 이후값으로 padding

    # 일단위 -> 시간단위
    oil_h = pd.DataFrame(columns=oil.columns)
    for i in range(oil.shape[1]):
        tmp = np.ravel(np.repeat(oil.iloc[:, i].values.reshape(-1, 1), 24, axis=1))
        oil_h[oil.columns[i]] = tmp
    date_range = pd.date_range(start, end, freq='H', closed='left')
    oil_h.index = date_range
    return oil_h

def load_dataset(path, start, end):
    smp_h = load_smp(path, start, end)
    power_h = load_pwr_sup_dem(path, start, end)
    oil_h = load_oil(path, start, end)
    data = pd.concat([smp_h, oil_h, power_h], axis=1)
    # time feature
    # data['date'] = data.index
    # data['year'] = data['date'].dt.year
    # data['month'] = data['date'].dt.month
    # data['day'] = data['date'].dt.day
    # data['dayofweek'] = data['date'].dt.dayofweek
    return data

def trans(dataset, pasts, future, x_col_index, y_col_index):
    pasts_rev = np.insert(pasts+1, 0, 0)
    data_agg = np.zeros((dataset.shape[0]-pasts.max()-future,pasts.sum()+len(pasts)))
    labels = np.zeros((dataset.shape[0]-pasts.max()-future, future))
    strat, end = 0, dataset.shape[0]
    for j, x_col in enumerate(x_col_index):
        strat = strat + pasts[j]
        data = []
        dataset_sub = dataset[:, x_col]
        for i in range(strat, end - future):
            indices = np.array(dataset_sub[i - pasts[j]:i+1])
            data.append(indices)
        data = np.array(data)
        data = data.reshape(data.shape[0], -1)
        data = data[max(pasts) - pasts[j]:, :]
        data_agg[:,pasts_rev[:j+1].sum():pasts_rev[:j+2].sum()] = data
        strat = 0
    for j, i in enumerate(range(max(pasts), end - future)):
        labels[j,:] = np.array(dataset[i+1:i + future+1, y_col_index])
    return data_agg, labels

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param

def save_obj(obj, name):
    try:
        with open('results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        os.mkdir('results')
        with open('results/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size
  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])
    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])
  return np.array(data), np.array(labels)

def create_time_steps(length):
  return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction, STEP):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)**0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'mae: {mae}, mse: {mse}, rmse: {rmse}, mape: {mape}')
    return mae, mse, rmse, mape