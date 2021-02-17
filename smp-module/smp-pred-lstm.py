from util import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras import initializers
import warnings
warnings.filterwarnings("ignore")
import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)

#%% load dataset
path_home = 'data/'
start = '2016-04-01'
end = '2018-04-01'
df_m = load_dataset(path_home, start, end)
df_u = df_m.iloc[:,[0]]

df_s = df_m.iloc[:,0:5].copy()

# scaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
df_s = scaler.fit_transform(df_s)

#%% make train dataset
FEATURES = df_s.shape[1]
past_history = 24
future_target = 24
STEP = 1
BATCH_SIZE = 128
EPOCHS = 300
EVALUATION_INTERVAL = 3
BUFFER_SIZE = 1000
TRAIN_SPLIT = int(df_u.shape[0] * 0.8)

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
  data = np.array(data)
  labels = np.array(labels)
  if FEATURES == 1:
    # univariate
    data = data.reshape(-1,history_size,1)
  return data, labels

# transformation
dataset = df_s
x_train, y_train = multivariate_data(dataset, dataset[:, 0], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=False)

x_val, y_val = multivariate_data(dataset, dataset[:, 0],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=False)


#
# numpy to tensor
# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# # train_data = train_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
# train_data = train_data.batch(BATCH_SIZE)
# train_data = train_data.repeat(EPOCHS)
# val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_data = val_data.batch(BATCH_SIZE).repeat()

# lr scheduler
def scheduler(epoch, lr):
    if epoch < 50:
        return 0.01
    else:
        return 0.01 * tf.math.abs((1 - epoch/(EPOCHS+1)))

lr = LearningRateScheduler(scheduler)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15,restore_best_weights=True)

# model train
model = Sequential([
    tf.keras.layers.LSTM(24*3,
                         return_sequences=False,
                         input_shape=x_train.shape[-2:],
                         kernel_initializer=initializers.he_normal()),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(future_target*3,
                          kernel_initializer=initializers.he_normal()),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(future_target,
                          kernel_initializer=initializers.he_normal()),
])

model.compile(optimizer='adam', loss='mape', metrics=['mape', 'mae'])
history = model.fit(x_train, y_train, epochs=EPOCHS,validation_data=(x_val, y_val), callbacks=[lr])

tf.keras.backend.clear_session()


#%% plot history
def plot_history(histories, key='acc'):
    plt.figure()

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

        # idx = np.argmin(history.history['val_' + key])
        idx = len(history.history['loss']) - 1
        best_tr = history.history[key][idx]
        best_val = history.history['val_' + key][idx]

        print('Train {} is {:.3f}, Val {} is {:.3f}'.format(key, best_tr, key, best_val))

    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.legend()
    plt.ylim([3, 6])
    plt.title('Result of training')
    plt.show()

plot_history([('model #10', history)],key='mape')


#%% model load
model = Sequential([
    tf.keras.layers.LSTM(672,
                         return_sequences=False,
                         input_shape=x_train.shape[-2:],
                         kernel_initializer=initializers.he_normal()),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(672,
                          activation='relu',
                          kernel_initializer=initializers.he_normal()),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(future_target,
                          activation='relu',
                          kernel_initializer=initializers.he_normal())
])

model.load_weights('models/results_0731/lstm_0731')

#%% test
def make_time_range(start, end):
    return pd.date_range(start, end, freq='H', closed='left')

def transform(df, start, end, is_label=False):
    _range = make_time_range(start, end)
    if is_label:
        _test_df = df.loc[_range[0]:_range[-1],'smp']
        _test_arr = _test_df.values.reshape(1, -1)
    else:
        _test_df = df.loc[_range[0]:_range[-1],'smp']
        _test_arr = _test_df.values.reshape(1, -1, FEATURES)
    return _range, _test_arr

start = '2018-03-01'
end = '2018-05-29'
df_test = load_dataset(path_home, start, end)
df_test = df_test.iloc[:,[0]]

# 예측1: 3월로 4월 예측
x_range, x_test = transform(df = df_test, start = '2018-03-04', end = '2018-04-01')
y_range, y_test = transform(df = df_test, start = '2018-04-01', end = '2018-04-29', is_label=True)
y_pred_4 = model.predict(x_test)

evaluate(y_test, y_pred_4)
plt.plot(x_range, np.ravel(x_test))
plt.plot(y_range, np.ravel(y_pred_4), label='predicted')
plt.plot(y_range, np.ravel(y_test), label = 'true')
plt.legend()
plt.xticks(rotation=25)
plt.show()

# 예측2: 4월로 5월 예측
x_range, x_test = transform(df = df_test, start = '2018-04-03', end = '2018-05-01')
y_range, y_test = transform(df = df_test, start = '2018-05-01', end = '2018-05-29', is_label=True)
y_pred_5 = model.predict(x_test)

plt.plot(x_range, x_test.reshape(-1,1)[:,0])
plt.plot(y_range, np.ravel(y_pred_5), label='predicted')
plt.plot(y_range, np.ravel(y_test), label = 'true')
plt.legend()
plt.xticks(rotation=25)
plt.show()
