import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def load_smp(path, start, end):
    date_range = pd.date_range(start, end, freq='D', closed='left')
    smp_raw = pd.read_csv(path + '전력가격/SMP.csv')
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


#%% load dataset
path_home = 'data/'
start = '2012-06-01'
end = '2018-06-01'
df = load_smp(path_home, start, end)
df['date'] = df.index

start = df.shape[0] - 24*28
end = df.shape[0]-1
steps = end - start + 1
train = df['smp'].values[:start]
test = df['smp'].values[start:]

from statsmodels.tsa.arima_model import ARIMA
model_arima = ARIMA(train,order=(3, 1, 1))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)
predictions = model_arima_fit.forecast(steps=steps)[0]
plt.plot(df['date'][start-24*28:start],train[-24*28:], label='history')
plt.plot(df['date'][start:],test, label='true')
plt.plot(df['date'][start:], predictions, label='pred')
plt.xticks(rotation=25)
plt.legend()
plt.show()