# preparation and generating synthetic data
import matplotlib.pyplot as plt
import numpy as np

def plot_series(time, series, start=0, end=None, format="-"):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend(["series"])
    plt.grid(False)

def trend(time, slope):
    return time * slope

def seasonal_pattern(season_time):
    return np.where(season_time<0.1, np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, seed=None, noise_level=1):
    rdm = np.random.RandomState(seed)
    return rdm.randn(len(time)) * noise_level

time = np.arange(365*10+1, dtype="float32") #??why plus 1
baseline = 10
amplitude = 40
series = trend(time, 0.01) + baseline
series += seasonality(time, 365, amplitude=amplitude, phase=0)
series += noise(time, seed=42, noise_level=3)

plot_series(time, series)
plt.show()

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 2000
#%%
import tensorflow as tf
from tensorflow import keras


def windowed_dataset(series, window_size, shuffle_buffer_size, batch_size):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer_size).map(lambda window: (window[:-1], window[1:]))
    return dataset.batch(batch_size).prefetch(1)


tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

#batch_size = 32
dataset = windowed_dataset(x_train, window_size, shuffle_buffer_size, batch_size)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=5, padding="causal",
                        activation="relu", input_shape=[None, 1]),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x*100.0)
])
model.summary()

lr_scheduler = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20)
)
model.compile(loss=keras.losses.Huber(),
              optimizer=keras.optimizers.SGD(lr=1e-8, momentum=0.9),
              metrics=["mae"])
history = model.fit(dataset, epochs=100, verbose=2, callbacks=[lr_scheduler])
#%%
plt.semilogx(history.history["lr"], history.history["loss"])
plt.xlim((1e-8, 0.001))
plt.show()
#%%
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

batch_size = 64
dataset = windowed_dataset(x_train, window_size, shuffle_buffer_size, batch_size)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=5, padding="causal",
                        activation="relu", input_shape=[None, 1]),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x*100.0)
])


model.compile(loss=keras.losses.Huber(),
              optimizer=keras.optimizers.SGD(lr=5e-5, momentum=0.9),
              metrics=["mae"])
history = model.fit(dataset, epochs=500, verbose=2)
#%%
def forecast_dataset(model, series, window_size, batch_size):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)
    return forecast

forecast_series = forecast_dataset(model, series, window_size, batch_size)[:,-1,0]
forecast_valid = forecast_series[split_time-window_size:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, forecast_valid)
plt.show()
#%%
tf.keras.losses.mean_absolute_error(x_valid, forecast_valid).numpy()
#%%
loss = history.history["loss"]
mae = history.history["mae"]
epochs = range(len(loss))
plt.plot(epochs, loss, label="loss")
plt.plot(epochs, mae, label="mae")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.figure()

loss_zoom = loss[200:]
mae_zoom = mae[200:]
epochs_zoom = epochs[200:]
plt.plot(epochs_zoom, loss_zoom, label="loss")
plt.plot(epochs_zoom, mae_zoom, label="mae")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.figure()
plt.show()