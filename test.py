import warnings
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K
import tensorflow as tf
from etc import Model_utils_2
from sklearn.preprocessing import StandardScaler
from utils import *

# tf.compat.v1.global_variables
sns.set()
warnings.filterwarnings("ignore")
print(tf.test.gpu_device_name())

DQN = tf.keras.models.load_model('models/multi_model.hdf5')
reward_save_dir = 'results/0817/'
DQN_ts = tf.keras.models.load_model('models/TS_multi_model.hdf5')

#%%
start = '2018-04-01'
end = '2018-04-29'

# DQN hyperparameters
state_size = 5  # PV, load, SMP, past 24 hour average SMP, SOC, acc_load
action_size = 9
learning_rate = 0.001

# Training hyperparameters
timesteps = 24*28  # 24시간 1일

# Exploration hyperparameters for epsilon greedy strategy
explore_start = 1.  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob

# Q-learning hyperparameters
gamma = 0.999  # Discounting rate of future reward

# Memory hyperparameters
pretrain_length = 10000  # # of experiences stored in Memory during initialization
memory_size = 10000  # # of experiences Memory can keep

#%%
# path = 'Final Modified Data.csv'
df = pd.read_csv('data/state_pred.csv')
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
if start == None:
    start = df['date'][0]
if end == None:
    end = df['date'][-1]
date_range = pd.date_range(start, end, freq='H', closed='left')
df = df.loc[date_range[0]:date_range[-1],:]
time = df['date']
del df['date']
df = df.reset_index(drop=True)

no_usage = 8  # 한달 8번 세탁기 사용
test_duration = 24*28  # 최근 1달 테스트
df = df[len(df) - test_duration:len(df)]
df = df.reset_index()
del df['index']

"""**Data Preprocessing_Standardization**"""
# MinMaxScaler
# The mean is not shifted to zero-centered

df_pv = df.iloc[:, 2:].values
df_load = df.iloc[:, 1:2].values
df_price = df.iloc[:, 0:1].values

# scale adjustment
df_pv = df_pv * 0.015
# df_load = df_load * 0.1

sc_price = StandardScaler(with_mean=False)
sc_energy = StandardScaler(with_mean=False)

pv = sc_energy.fit_transform(df_pv)
load = sc_energy.transform(df_load)
price = sc_price.fit_transform(df_price)
# acc_load = np.cumsum(load)

x = np.concatenate([pv, load, price], axis=-1)

"""
**Hyperparameters Setting**
"""

appl = Appliance(action_size=1)

battery = Model_utils_2.Battery(action_size=action_size,
                                scaler_energy=sc_energy,
                                scaler_price=sc_price)
memory = Model_utils_2.Memory(memory_size)
memory_ts = Model_utils_2.Memory(memory_size)
appl = Appliance(action_size=1)
battery = Model_utils_2.Battery(action_size=action_size,
                                scaler_energy=sc_energy,
                                scaler_price=sc_price)

SOC = np.array([battery.initial_SOC])
historical_price = np.zeros(timesteps)
day = 0
hour = 0
timestep = 0
ts_stack = 0
done = False
pv_list = []
load_list = []
price_list = []
SOC_list = []
fault_list = []
ts_action_list = []
ts_reward_list = []
ns_action_list = []
ns_reward_list = []

av_price_list = []
trading_price_list = []
x[:, 1] = load[:, 0]

while day < len(x) / 24:
    historical_price[timestep] = x[day * 24 + hour, 2]
    average_price = np.mean(np.array([price for price in historical_price if price != 0]))

    ts_state = np.concatenate(
        (x[day * 24 + hour, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour / 24])), axis=-1)
    ts_action = np.argmax(DQN_ts.predict(np.expand_dims(ts_state, axis=0)))
    ts_stack, ts_load, ts_reward = appl.compute(ts_state, ts_action)
    x[day * 24 + hour, 1] += ts_load

    state = np.concatenate((x[day * 24 + hour, :], SOC, np.array([average_price])), axis=-1)
    action = np.argmax(DQN.predict(np.expand_dims(state, axis=0)))

    next_SOC, ns_reward, state_update, comsumed_price, trading_price = battery.compute(state, action,
                                                                                       day * 24 + hour)  # 보상 얻는 텀
    trading_price_list.append(trading_price)

    pv_list.append(state_update[0])
    load_list.append(state_update[1])
    price_list.append(state_update[2])
    SOC_list.append(state_update[3])
    av_price_list.append(state_update[4])

    SOC = next_SOC
    ts_reward_list.append(ts_reward)
    ts_action_list.append(ts_action)
    ns_reward_list.append(ns_reward)
    ns_action_list.append(action)

    if hour < 23:
        hour += 1
        timestep += 1
        if timestep >= timesteps:  # timesteps=24
            timestep = 0

        historical_price[timestep] = x[day * 24 + hour, 2]
        average_price = np.mean(np.array([price for price in historical_price if price != 0]))
        ts_next_state = np.concatenate(
            (x[day * 24 + hour, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour / 24])),
            axis=-1)
        # accumulated_load = np.sum(load_list, axis=0)
        next_state = np.concatenate((x[day * 24 + hour, :], next_SOC, np.array([average_price])), axis=-1)
    else:
        done = True
        day += 1
        hour = 0
        timestep += 1
        if timestep >= timesteps:
            timestep = 0
        if day < len(x) / 24:
            historical_price[timestep] = x[day * 24 + hour, 2]
            average_price = np.mean(np.array([price for price in historical_price if price != 0]))
            ts_next_state = np.concatenate(
                (x[day * 24 + hour, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour / 24])),
                axis=-1)
            # accumulated_load = np.sum(load_list, axis=0)
            next_state = np.concatenate(
                (x[day * 24 + hour, :], next_SOC, np.array([average_price])), axis=-1)
        else:
            break


#%% plotting
def Tax_fn_true(acc_load): #unit[kW/h]
    # 한달이 끝났을 때 예상 load 계산
    thres_q1 =200 #[W/day]
    thres_q2 = 400
    price_sum = 0
    if acc_load < thres_q1:
        price_sum +=acc_load * 93.3
    elif acc_load < thres_q1 + thres_q2:
        price_sum += thres_q1 * 93.3
        price_sum += (acc_load - thres_q1)* 187.9
    else:
        price_sum += thres_q1 * 93.3
        price_sum += (acc_load - thres_q1) * 187.9
        price_sum += (acc_load - thres_q1 - thres_q2) * 280.6
    return price_sum

tax_true = Tax_fn_true(np.sum(df_load) + 0.5 * 8)

# tax = Tax_fn_total(np.sum(load_list), sc_energy)
trading_price_all = np.sum(trading_price_list)

#%% scheduling
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
font = {'size': 15}
mpl.rc('font', **font)

is_sell = (np.array(ns_action_list) < 4).astype(int)
is_self = ((np.array(ns_action_list) >= 4) * (np.array(ns_action_list) <= 7)).astype(int)
is_wash = (np.array(ts_action_list) == 1).astype(int)

fig, ax = plt.subplots(figsize = (25,5))
for i in range(np.where(is_self)[0].shape[0]):
    if i != np.where(is_self)[0].shape[0] - 1:
        ax.fill_between([np.where(is_self)[0][i],np.where(is_self)[0][i]+1], 0, 1,
                        color='g', alpha=0.2, transform=ax.get_xaxis_transform())
    else:
        ax.fill_between([np.where(is_self)[0][i], np.where(is_self)[0][i] + 1], 0, 1,
                        color='g', alpha=0.2, transform=ax.get_xaxis_transform(), label='self consumption')
for i in range(np.where(is_sell)[0].shape[0]):
    if i != np.where(is_sell)[0].shape[0] -1:
        ax.fill_between([np.where(is_sell)[0][i],np.where(is_sell)[0][i]+1], 0, 1,
                        color='y', alpha=0.2, transform=ax.get_xaxis_transform())
    else:
        ax.fill_between([np.where(is_sell)[0][i], np.where(is_sell)[0][i] + 1], 0, 1,
                        color='y', alpha=0.2, transform=ax.get_xaxis_transform(), label = 'sell')

for i in range(np.where(is_wash)[0].shape[0]):
    if i != np.where(is_wash)[0].shape[0] - 1:
        ax.fill_between([np.where(is_wash)[0][i],np.where(is_wash)[0][i]+1], 0, 1,
                        color='b', alpha=0.2, transform=ax.get_xaxis_transform())
    else:
        ax.fill_between([np.where(is_wash)[0][i], np.where(is_wash)[0][i] + 1], 0, 1,
                        color='b', alpha=0.2, transform=ax.get_xaxis_transform(), label='appliance')

plt.plot(SOC_list, "ro-", label = "SOC")
plt.plot(pv_list, "g", label = "pv")
plt.plot(load_list, "m", label = "load")
plt.plot(price_list, "bs", label = "price")
plt.plot(av_price_list, label = "av_price")
# plt.bar(range(0, 24 * i), battery.action_set[action_list[-24 * i:]],
#         facecolor = "w", label = "action")
plt.ylabel("SOC/ Normalized Price")
plt.xlabel("Hour")
plt.legend()
plt.show()

#%% total reward
reward_save_dir = 'results/'
with open (reward_save_dir+'ts_reward', "rb") as file: #로드: 한줄씩 파일 읽어옴
  ts_reward_list = pickle.load(file)
with open (reward_save_dir+'ns_reward', "rb") as file: #로드: 한줄씩 파일 읽어옴
  ns_reward_list = pickle.load(file)

plt.plot(ts_reward_list, label='appliance')
plt.plot(ns_reward_list, label='sell agent')
plt.xlabel('# episode')
plt.ylabel('reward')
plt.legend()
plt.show()


#%% 수익
with open (reward_save_dir+'trading_price_list_all', "rb") as file: #로드: 한줄씩 파일 읽어옴
  trading_price_list_all = pickle.load(file)
with open (reward_save_dir+'tax_list', "rb") as file: #로드: 한줄씩 파일 읽어옴
  tax_list = pickle.load(file)

plt.plot(trading_price_list_all, label='appliance')
plt.plot(tax_list, label='sell agent')
plt.plot(np.array(tax_list) - np.array(trading_price_list_all), label='Bills')
plt.xlabel('# episode')
plt.ylabel('Price [won]')
plt.legend()
plt.show()

#%% action
Tax_fn_true(np.sum(df_load) + (0.5 * 8))
Tax_fn_true(sc_energy.inverse_transform(np.array(load_list)).sum())

is_self = (np.array(ns_action_list) < 4).astype(int)
is_sell = ((np.array(ns_action_list) >= 4) * (np.array(ns_action_list) <= 7)).astype(int)
is_wash = (np.array(ts_action_list) == 1).astype(int)

df_action = pd.DataFrame(columns=['self','sell','wash'])
df_action['self'] = is_self
df_action['sell'] = is_sell
df_action['wash'] = is_wash
df_action.index = time
# df_action.to_csv('action_result_5.csv',index=True)

np.sum(load_list)
Tax_fn_true(Model_utils_2.inverse_transform(sc_energy, load.sum()) + 0.5 * 8)

Tax_fn_true(Model_utils_2.inverse_transform(sc_energy, np.sum(load_list)))

#%% evaluation
def Tax_fn_true(acc_load): #unit[kW/h]
    # 한달이 끝났을 때 예상 load 계산
    thres_q1 =200 #[W/day]
    thres_q2 = 400
    price_sum = 0
    if acc_load < thres_q1:
        price_sum +=acc_load * 93.3
    elif acc_load < thres_q1 + thres_q2:
        price_sum += thres_q1 * 93.3
        price_sum += (acc_load - thres_q1)* 187.9
    else:
        price_sum += thres_q1 * 93.3
        price_sum += (acc_load - thres_q1) * 187.9
        price_sum += (acc_load - thres_q1 - thres_q2) * 280.6
    return price_sum


def evaluation(scaled_load_list, trading_price_list, raw_load, raw_smp, raw_pv, ts_flag =True, ts_num_usage=9):
    # load list를 inverse transform
    scheduled_load = sc_energy.inverse_transform(np.array(scaled_load_list)).sum() + ts_flag * 0.5 * ts_num_usage
    # 거래 이익
    trading_price = np.sum(trading_price_list)
    print('1. 스케줄링 한 결과 (acc_load 미포함)')
    print('예상 세금: {:.2f}, 예상 판매수익: {:.2f}, 예상 지불:{:.2f}'.format(Tax_fn_true(scheduled_load),trading_price,Tax_fn_true(scheduled_load)-trading_price))
    print('===============================================================================')
    print('2. 스케쥴링 안 한 결과')
    # reference
    ref_load = raw_load.sum() + ts_num_usage * 0.5
    ref_trading_price = raw_smp.mean() * raw_pv.sum()
    print('예상 세금: {:.2f}, 예상 판매 수익 (평균 smp로 pv 팔았을 시): {:.2f}, '
          '예상 지불 1: {:.2f}, 예상 지불 2 (pv 팔지 않고 자가발전): {:.2f}'.format(Tax_fn_true(ref_load),ref_trading_price,
                                                                         Tax_fn_true(ref_load) - ref_trading_price,
                                                                         Tax_fn_true(ref_load - raw_pv.sum())))

evaluation(load_list, trading_price_list, df_load,df_price, df_pv, ts_flag = False)
