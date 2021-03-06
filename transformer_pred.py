import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import *
from keras.layers import *

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from binance_data import get_binance_data


class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, seq_len)'''
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        x = tf.math.reduce_mean(x[:, :, :4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear  # Linear time feature
        time_linear = tf.expand_dims(time_linear, axis=-1)  # Add dimension (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)  # Add dimension (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1)  # shape = (batch, seq_len, 2)

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config


class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

        self.key = Dense(self.d_k,
                         input_shape=input_shape,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')

        self.value = Dense(self.d_v,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x / np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out

    #############################################################################


class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))

            # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
        self.linear = Dense(input_shape[0][-1],
                            input_shape=input_shape,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear

    #############################################################################


class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1)
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config

if __name__ == '__main__':
    batch_size = 32
    seq_len = 80

    d_k = 160
    d_v = 160
    n_heads = 12
    ff_dim = 160

    df = pd.DataFrame(get_binance_data(True), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    # Replace 0 to avoid dividing by 0 later on
    df['Volume'].replace(to_replace=0, method='ffill', inplace=True)
    df["Date"] = pd.to_datetime(df['Date'], unit='ms')

    df.sort_values('Date', inplace=True)
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].rolling(5).mean()

    # Drop all rows with NaN values
    df['Open'] = pd.to_numeric(df['Open']).pct_change() # Create arithmetic returns column
    df['High'] = pd.to_numeric(df['High']).pct_change() # Create arithmetic returns column
    df['Low'] = pd.to_numeric(df['Low']).pct_change() # Create arithmetic returns column
    df['Close'] = pd.to_numeric(df['Close']).pct_change() # Create arithmetic returns column
    df['Volume'] = pd.to_numeric(df['Volume']).pct_change() # Create arithmetic returns column

    df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

    ###############################################################################
    '''Create indexes to split dataset'''

    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

    ###############################################################################
    '''Normalize price columns'''
    #
    min_return = min(df[(df.index < last_20pct)][['Open', 'High', 'Low', 'Close']].min(axis=0))
    max_return = max(df[(df.index < last_20pct)][['Open', 'High', 'Low', 'Close']].max(axis=0))

    # Min-max normalize price columns (0-1 range)
    df['Open'] = (df['Open'] - min_return) / (max_return - min_return)
    df['High'] = (df['High'] - min_return) / (max_return - min_return)
    df['Low'] = (df['Low'] - min_return) / (max_return - min_return)
    df['Close'] = (df['Close'] - min_return) / (max_return - min_return)

    ###############################################################################
    '''Normalize volume column'''

    min_volume = df[(df.index < last_20pct)]['Volume'].min(axis=0)
    max_volume = df[(df.index < last_20pct)]['Volume'].max(axis=0)

    # Min-max normalize volume columns (0-1 range)
    df['Volume'] = (df['Volume'] - min_volume) / (max_volume - min_volume)

    ###############################################################################
    '''Create training, validation and test split'''

    df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]

    # Remove date column
    df_train.drop(columns=['Date'], inplace=True)
    df_val.drop(columns=['Date'], inplace=True)
    df_test.drop(columns=['Date'], inplace=True)

    # Convert pandas columns into arrays
    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values
    print('Training data shape: {}'.format(train_data.shape))
    print('Validation data shape: {}'.format(val_data.shape))
    print('Test data shape: {}'.format(test_data.shape))

    # Training data
    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        X_train.append(train_data[i - seq_len:i])  # Chunks of training data with a length of 128 df-rows
        y_train.append(train_data[:, 3][i])  # Value of 4th column (Close Price) of df-row 128+1
    X_train, y_train = np.array(X_train), np.array(y_train)

    ###############################################################################

    # Validation data
    X_val, y_val = [], []
    for i in range(seq_len, len(val_data)):
        X_val.append(val_data[i - seq_len:i])
        y_val.append(val_data[:, 3][i])
    X_val, y_val = np.array(X_val), np.array(y_val)

    ###############################################################################

    # Test data
    X_test, y_test = [], []
    for i in range(seq_len, len(test_data)):
        X_test.append(test_data[i - seq_len:i])
        y_test.append(test_data[:, 3][i])
    X_test, y_test = np.array(X_test), np.array(y_test)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)


    def create_model():
        '''Initialize time and transformer layers'''
        time_embedding = Time2Vector(seq_len)
        attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

        '''Construct model'''
        in_seq = Input(shape=(seq_len, 5))
        x = time_embedding(in_seq)
        x = Concatenate(axis=-1)([in_seq, x])
        x = attn_layer1((x, x, x))
        x = attn_layer2((x, x, x))
        x = attn_layer3((x, x, x))
        x = GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        out = Dense(1, activation='linear')(x)

        model = Model(inputs=in_seq, outputs=out)
        model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
        return model


    model = create_model()
    model.summary()

    callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5',
                                                  monitor='val_loss',
                                                  save_best_only=True,
                                                  verbose=1)

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=3,
                        callbacks=[callback],
                        validation_data=(X_val, y_val))

    model = tf.keras.models.load_model('Transformer+TimeEmbedding.hdf5',
                                       custom_objects={'Time2Vector': Time2Vector,
                                                       'SingleAttention': SingleAttention,
                                                       'MultiAttention': MultiAttention,
                                                       'TransformerEncoder': TransformerEncoder})

    ###############################################################################
    '''Calculate predictions and metrics'''

    # Calculate predication for training, validation and test data
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Print evaluation metrics for all datasets
    train_eval = model.evaluate(X_train, y_train, verbose=0)
    val_eval = model.evaluate(X_val, y_val, verbose=0)
    test_eval = model.evaluate(X_test, y_test, verbose=0)
    print(' ')
    print('Evaluation metrics')
    print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
    print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
    print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

    ###############################################################################
    '''Display results'''

    fig = plt.figure(figsize=(15, 20))
    st = fig.suptitle("Moving Average - Transformer + TimeEmbedding Model", fontsize=22)
    st.set_y(0.92)

    # Plot training data results
    ax11 = fig.add_subplot(311)
    ax11.plot(train_data[:, 3], label='IBM Closing Returns')
    ax11.plot(np.arange(seq_len, train_pred.shape[0] + seq_len), train_pred, linewidth=3,
              label='Predicted IBM Closing Returns')
    ax11.set_title("Training Data", fontsize=18)
    ax11.set_xlabel('Date')
    ax11.set_ylabel('IBM Closing Returns')
    ax11.legend(loc="best", fontsize=12)

    # Plot validation data results
    ax21 = fig.add_subplot(312)
    ax21.plot(val_data[:, 3], label='IBM Closing Returns')
    ax21.plot(np.arange(seq_len, val_pred.shape[0] + seq_len), val_pred, linewidth=3,
              label='Predicted IBM Closing Returns')
    ax21.set_title("Validation Data", fontsize=18)
    ax21.set_xlabel('Date')
    ax21.set_ylabel('IBM Closing Returns')
    ax21.legend(loc="best", fontsize=12)

    # Plot test data results
    ax31 = fig.add_subplot(313)
    ax31.plot(test_data[:, 3], label='IBM Closing Returns')
    ax31.plot(np.arange(seq_len, test_pred.shape[0] + seq_len), test_pred, linewidth=3,
              label='Predicted IBM Closing Returns')
    ax31.set_title("Test Data", fontsize=18)
    ax31.set_xlabel('Date')
    ax31.set_ylabel('IBM Closing Returns')
    ax31.legend(loc="best", fontsize=12)

    plt.show()
