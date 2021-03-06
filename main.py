# -*- coding: utf-8 -*-
import utils
from utils import *

sns.set()
warnings.filterwarnings("ignore")
print(tf.test.gpu_device_name())

#%% user input
cont = False
start = '2018-04-01'
end = '2018-04-29'
# start = None
# end = '2018-04-01'

# DQN hyperparameters
state_size = 6  # PV, load, SMP, past 24 hour average SMP, SOC, acc_load
action_size = 9
learning_rate = 0.001

# Training hyperparameters
episodes = 100
batch_size = 24
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

reward_save_dir = 'results/'
save_model_name = 'multi_0817.hdf5'

#%% Initialize
"""**Load Data**"""
path = 'data/state_true.csv'

df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
if start == None:
    start = df['date'][0]
if end == None:
    end = df['date'][-1]
date_range = pd.date_range(start, end, freq='H', closed='left')
df = df.loc[date_range[0]:date_range[-1],:]
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
df_load = df_load * 0.05

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
battery = utils.Battery(action_size=action_size,
                        scaler_energy=sc_energy,
                        scaler_price=sc_price)
memory = utils.Memory(memory_size)
memory_ts = utils.Memory(memory_size)

DQN = utils.DQNNet(state_size=state_size, action_size=action_size, learning_rate=learning_rate)
DQN_ts = utils.DQNNet_ts(state_size=4, action_size=2, learning_rate=learning_rate)

"""**Memory Initialization**"""
memory = utils.Memory_Initialization(memory, battery, timesteps, pretrain_length, x, action_size, sc_energy)
memory_ts = utils.TS_Memory_Initialization(memory_ts, appl, timesteps, pretrain_length, x, 2)

"""**DQN Training**"""
chk_lst = []

decay_step = 0  # Decay rate for ϵ-greedy policy
total_reward_bef = -10e8

# 필요한 것 저장
total_reward_list = []
update_load_list = []
ts_action_list = np.zeros((episodes,24*28))
ts_reward_list = np.zeros((episodes,24*28))
ns_action_list = np.zeros((episodes,24*28))
ns_reward_list = np.zeros((episodes,24*28))

option = 2 # 1 지훈오빠 2 나

trading_price_list_all = []
tax_list_all = []

#%% Train
for e in range(0, episodes, 1):
    "Section 1: Env.reset()"
    total_reward = 0
    total_reward_ns = 0
    total_reward_ts = 0
    tax = 0
    SOC = np.array([battery.initial_SOC])
    historical_price = np.zeros(timesteps)
    day = 0
    hour = 0
    timestep = 0
    done = False
    ts_stack = 0
    acc_load = 0
    trading_price_list = []
    # load 초기화
    x[:, 1] = load[:, 0]
    # 하루에 한번 한시간 조건 걸어야함
    if option == 1:
        load = utils.usage_appliance(no_usage, test_duration, load)  # Usage of appliance: 500W, 에피소드마다 업데이트
        # check_updated load
        update_load_list.append(load)

    # state_list - load update
    load_list = []
    load_list.append(0)
    SOC_list = []
    # load_list.append(0)
    "Section 2: time_steps"
    while day < len(x) / 24:  # while문 하루씩 진행, input의 날짜크기 * 24만큼 돔
        # 지난 24시간의 전기요금 추적
        historical_price[timestep] = x[day * 24 + hour, 2]  # 1시간씩 저장
        average_price = np.mean(np.array([price for price in historical_price if price != 0]))

        if option == 2:
            """TS"""
            ts_state = np.concatenate((x[day * 24 + hour, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour/24])), axis=-1)
            # ϵ-greedy policy
            exp_exp_tradeoff = np.random.rand()
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
            if (explore_probability > exp_exp_tradeoff):
                ts_action = np.random.randint(0, 2)
            else:
                if cont:
                    ts_action = np.argmax(DQN_ts.predict(np.expand_dims(ts_state, axis=0)))
                else:
                    ts_action = np.argmax(DQN_ts.model.predict(np.expand_dims(ts_state, axis=0)))
            ts_stack, ts_load, ts_reward = appl.compute(ts_state, ts_action)
            x[day * 24 + hour, 1] += ts_load
            update_load_list.append(ts_load)
            ts_action_list[e, day*24+hour] = ts_action
            ts_reward_list[e, day*24+hour] = ts_reward
            total_reward_ts += ts_reward

        """NS"""
        # state: PV, Load, Price, SOC, Avg.Price, accumulated load - 6개
        # acc_load = x[:day * 24 + hour+1,1].sum() + ts_action_list[e, :day*24+hour+1].sum()
        # acc_load = sc_energy.transform(acc_load)
        accumulated_load = np.cumsum(load_list)
        if hour == 0:
            state = np.concatenate((x[day * 24 + hour, :], SOC, np.array([average_price]), np.array([accumulated_load[day*24]])), axis=-1)
        else:
            state = np.concatenate(
                (x[day * 24 + hour, :], SOC, np.array([average_price]), np.array([0])), axis=-1)
        # state = np.concatenate((x[day * 24 + hour, :], SOC, np.array([average_price])), axis=-1)

        # ϵ-greedy policy
        exp_exp_tradeoff = np.random.rand()
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        if (explore_probability > exp_exp_tradeoff):
            action = np.random.randint(0, action_size)
        else:
            if cont:
                action = np.argmax(DQN.predict(np.expand_dims(state, axis=0)))
            else:
                action = np.argmax(DQN.model.predict(np.expand_dims(state, axis=0)))

        # Compute the reward and new state based on the selected action
        next_SOC, ns_reward, state_update, comsumed_price, trading_price = battery.compute(state, action, day*24+hour)  # 보상 얻는 텀
        total_reward_ns += ns_reward
        ns_action_list[e, day*24+hour] = action
        ns_reward_list[e, day * 24 + hour] = ns_reward

        # load updated for evaluation
        # consumed_price_list.append(comsumed_price)
        trading_price_list.append(trading_price)
        load_list.append(state_update[1])
        SOC_list.append(state_update[3])

        """memory"""
        # Store the experience in memory
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
            accumulated_load = np.cumsum(load_list)
            next_state = np.concatenate(
                (x[day * 24 + hour, :], next_SOC, np.array([average_price]), np.array([0])),
                axis=-1)
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
                accumulated_load = np.cumsum(load_list)
                next_state = np.concatenate(
                    (x[day * 24 + hour, :], next_SOC, np.array([average_price]), np.array([accumulated_load[day * 24]])),
                    axis=-1)
            else:
                break

        SOC = next_SOC
        ts_experience = ts_state, ts_action, ts_reward, ts_next_state, done
        memory_ts.store(ts_experience)

        experience = state, action, ns_reward, next_state, done
        memory.store(experience)
        decay_step += 1

        # DQN training - minibatch
        memory = utils.DQN_minibatch(memory, DQN, batch_size, cont, gamma, action_size)
        memory_ts = utils.DQN_minibatch(memory_ts, DQN_ts, batch_size, cont, gamma, 2)

    total_reward = total_reward_ts + total_reward_ns

    "Section 4: 월말 최종 reward 평가"
    total_reward_list.append(total_reward)
    if total_reward > total_reward_bef:
        total_reward_bef = total_reward
        # drive.mount('/content/gdrive')
        # DQN.model.save_weights(directory + "RL_multi_0813.hdf5")
        if cont:
            DQN.save('models/' + save_model_name)
            DQN_ts.save('models/TS_' + save_model_name)
        else:
            DQN.model.save('models/'+save_model_name)
            DQN_ts.model.save('models/TS_'+save_model_name)
    tax = utils.Tax_fn_total(np.sum(load_list), sc_energy)
    print(f'load list is {load_list}')
    print(f'Tax price is {tax}')
    print(f'Trading gain is {np.sum(trading_price_list)}')
    print(f'Total is {-tax + np.sum(trading_price_list)}')
    "Section 4"
    print("Episode: {}, ns_reward: {}, explore P: {:.2f}".format(e, total_reward_ns, explore_probability))
    print("Episode: {}, ts_reward: {}, explore P: {:.2f}".format(e, total_reward_ts, explore_probability))
    print("Episode: {}, total_reward: {}, explore P: {:.2f}".format(e, total_reward, explore_probability))
    print(f'Number of TS action is {np.sum(ts_action_list[e,:])}')

    trading_price_list_all.append(np.sum(trading_price_list))
    tax_list_all.append(tax)

#%%
""" Save info."""
with open(reward_save_dir + 'total_reward', "wb") as file:
    pickle.dump(total_reward_list, file)
with open(reward_save_dir+'ts_reward', "wb") as file:
    pickle.dump(ts_reward_list.sum(axis=1), file)
with open(reward_save_dir+'ns_reward', "wb") as file:
    pickle.dump(ns_reward_list.sum(axis=1), file)
with open(reward_save_dir+'tax_list', "wb") as file:
    pickle.dump(tax_list_all, file)
with open(reward_save_dir+'trading_price_list_all', "wb") as file:
    pickle.dump(trading_price_list_all, file)

#%% 결과 확인
# plt.plot(np.array(ts_action_list).sum(axis=1),'-')
# total reward 확인
plt.plot(total_reward_list, label = 'Total reward')
plt.plot(ts_reward_list.sum(axis=1), label = 'TS reward')
plt.plot(ns_reward_list.sum(axis=1), label = 'NS reward')
plt.legend()
plt.show()

plt.plot(tax_list_all)
plt.plot(trading_price_list_all)
plt.plot(np.array(tax_list_all) - np.array(trading_price_list_all))
plt.show()
