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
from keras.layers import LSTM, Dense, Dropout, SimpleRNN

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

regressor = Sequential()

# adding first RNN layer and dropout regulatization
regressor.add(
    SimpleRNN(units = 50,
              activation = "tanh",
              return_sequences = True,
              input_shape = (x_train_data.shape[1],1))
             )

regressor.add(
    Dropout(0.2)
             )


# adding second RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50,
              activation = "tanh",
              return_sequences = True)
             )

regressor.add(
    Dropout(0.2)
             )

# adding third RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50,
              activation = "tanh",
              return_sequences = True)
             )

regressor.add(
    Dropout(0.2)
             )

# adding fourth RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50)
             )

regressor.add(
    Dropout(0.2)
             )

# adding the output layer
regressor.add(Dense(units = 1))

# compiling RNN
regressor.compile(
    optimizer = "adam",
    loss = "mean_squared_error",
    metrics = ["accuracy"])

# fitting the RNN
regressor.fit(x_train_data, y_train_data, epochs = 1, batch_size = 1)

regressor.save("saved_rnn_model.h5")

