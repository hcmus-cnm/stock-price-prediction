import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams

from binance_data import get_binance_data

rcParams['figure.figsize'] = 20, 10

from sklearn.preprocessing import MinMaxScaler


df_nse = pd.DataFrame(get_binance_data(), columns=['ts', 'price'])
df_nse["time"] = pd.to_datetime(df_nse['ts'], unit='ms')
df_nse.index = df_nse['time']
df_nse.drop('ts', axis=1, inplace=True)

from keras.models import Sequential
from keras.layers import LSTM, Dense

data = df_nse.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df_nse)), columns=['Time', 'Close'])

for i in range(0, len(data)):
    new_dataset["Time"][i] = data['time'][i]
    new_dataset["Close"][i] = data["price"][i]

new_dataset.index = new_dataset.Time
new_dataset.drop("Time", axis=1, inplace=True)

final_dataset = new_dataset.values

train_data = final_dataset[:987, :]
valid_data = final_dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []

window_size = 80

for i in range(window_size, len(train_data)):
    x_train_data.append(scaled_data[i - window_size:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(window_size, inputs_data.shape[0]):
    X_test.append(inputs_data[i - window_size:i, 0])
X_test = np.array(X_test)


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = lstm_model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

lstm_model.save("saved_lstm_model.h5")

