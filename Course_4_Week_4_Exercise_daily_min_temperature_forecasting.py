import wget
import os
import csv

DOWNLOAD_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
daily_min_temp_dir = "D:/Code using Pycharm/Coursera course 4"
os.makedirs(daily_min_temp_dir, exist_ok=True)
csv_path = os.path.join(daily_min_temp_dir, "daily-min-temperatures.csv")
wget.download(DOWNLOAD_URL, out=csv_path)
#%%
import numpy as np

time_stamps = []
daily_min_temp = []
with open(csv_path) as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    next(reader)
    for row in reader:
        time_stamps.append(row[0])
        daily_min_temp.append(float(row[1]))
time_steps = range(len(time_stamps))

time = np.array(time_steps)
series = np.array(daily_min_temp, dtype="float32")
print("There are overall {} time steps".format(time.shape[0]))
#%%
import matplotlib.pyplot as plt
def plot_series(time, series, start=0, end=None, format="-"):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend(["series"])
    plt.grid(True)


plot_series(time, series)
#%%
import tensorflow as tf
from tensorflow import keras


split_time = 2500
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 50
batch_size = 32
shuffle_buffer_size = 2000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer_size).map(lambda window: (window[:-1], window[1:]))
    return dataset.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(32).prefetch(1)
    forecast = model.predict(dataset)
    return forecast

tf.random.set_seed(42)
np.random.seed(42)
tf.keras.backend.clear_session()

#window_size = 100
#batch_size = 64
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal",
                        input_shape=[None, 1]),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x*25.0)
])

lr_scheduler = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20)
)
model.compile(loss=keras.losses.Huber(),
              optimizer=keras.optimizers.SGD(lr=1e-8, momentum=.9),
              metrics=["mae"])
history = model.fit(dataset, epochs=100, verbose=2, callbacks=[lr_scheduler])
#%%
plt.semilogx(history.history["lr"], history.history["loss"])
plt.xlim((1e-8, 0.001))
plt.show()
#%%
tf.random.set_seed(42)
np.random.seed(42)
tf.keras.backend.clear_session()

window_size = 60
batch_size = 100
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal",
                        input_shape=[None, 1]),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True,
                                                 dropout=0.2, recurrent_dropout=0.2)),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True,
                                                 dropout=0.2, recurrent_dropout=0.2)),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x*25.0)
])

ckpt_dir = "D:/Code using Pycharm/Coursera course 4"
checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "my_dmt_model_1.h5"))
model.compile(loss=keras.losses.Huber(),
              optimizer=keras.optimizers.SGD(lr=7e-4, momentum=.9),
              metrics=["mae"])
history = model.fit(dataset, epochs=150, verbose=2, callbacks=[checkpoint_cb])

#%%
model = keras.models.load_model("my_dmt_model_1.h5")
forecast_series = model_forecast(model, series, window_size)
forecast_valid = np.array(forecast_series[split_time-window_size:-1, -1, 0])

plot_series(time_valid, x_valid)
plot_series(time_valid, forecast_valid)
plt.show()
#%%
tf.keras.losses.mean_absolute_error(x_valid, forecast_valid).numpy()
#%%
loss = history.history["loss"]
mae = history.history["mae"]
epochs = history.epoch

plt.plot(epochs, loss, label="loss")
plt.plot(epochs, mae, label="mae")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.grid(False)
plt.legend(["loss", "mae"])
plt.show()
plt.figure()
#%%
loss_zoom = loss[50:]
mae_zoom = mae[50:]
epochs_zoom = epochs[50:]
plt.plot(epochs_zoom, loss_zoom, label="loss")
plt.plot(epochs_zoom, mae_zoom, label="mae")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.grid(False)
plt.legend(["loss", "mae"])
plt.show()
plt.figure()
