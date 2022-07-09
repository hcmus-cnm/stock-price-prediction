
import dash
from dash import dcc, Output, Input, html
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from transformer_pred import Time2Vector, SingleAttention, MultiAttention, TransformerEncoder

app = dash.Dash()
server = app.server
window_size = 80

import pandas as pd
import numpy as np
from binance_data import get_binance_data

lstm_model = load_model("saved_lstm_model.h5")
rnn_model = load_model("saved_rnn_model.h5")

transform_model = load_model('Transformer+TimeEmbedding.hdf5',
                             custom_objects={'Time2Vector': Time2Vector,
                                             'SingleAttention': SingleAttention,
                                             'MultiAttention': MultiAttention,
                                             'TransformerEncoder': TransformerEncoder})


app.layout = html.Div([
    dcc.Interval(id='my-interval', interval=60_000),
    html.Div(id='my-output-interval'),
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    html.Div([
        html.H2("Actual closing price", style={"textAlign": "center"}),
        dcc.Graph(id="chart-data"),
    ])
])


@app.callback(
    Output('chart-data', 'figure'),
    [Input('my-interval', 'n_intervals')])
def display_output(n):

    full_data = pd.DataFrame(get_binance_data(True), columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    full_data["Time"] = pd.to_datetime(full_data['Time'], unit='ms')


    new_dataset = pd.DataFrame(index=range(0, len(full_data)), columns=['Time', 'Close'])

    for i in range(0, len(full_data)):
        new_dataset["Time"][i] = full_data['Time'][i]
        new_dataset["Close"][i] = full_data["Close"][i]
    # new_data = pd.DataFrame(get_binance_data(), columns=['Time', 'Close'])

    full_data['Open'] = pd.to_numeric(full_data['Open']).pct_change()  # Create arithmetic returns column
    full_data['High'] = pd.to_numeric(full_data['High']).pct_change()  # Create arithmetic returns column
    full_data['Low'] = pd.to_numeric(full_data['Low']).pct_change()  # Create arithmetic returns column
    full_data['Close'] = pd.to_numeric(full_data['Close']).pct_change()  # Create arithmetic returns column
    full_data['Volume'] = pd.to_numeric(full_data['Volume']).pct_change()  # Create arithmetic returns column

    full_data.drop(columns=['Time'], inplace=True)
    train_data = full_data.values

    X_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i - window_size:i])
    X_train = np.array(X_train)[-1280:]

    new_dataset.index = new_dataset.Time
    new_dataset.drop("Time", axis=1, inplace=True)
    dataset = new_dataset.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(dataset)

    inputs = new_dataset[window_size:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_input = np.array([
        inputs[i - window_size:i, 0]
        for i in range(window_size, inputs.shape[0])
    ])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    transform_closing_price = transform_model.predict(X_train)

    rnn_closing_price = rnn_model.predict(X_input)
    rnn_closing_price = scaler.inverse_transform(rnn_closing_price)

    lstm_closing_price = lstm_model.predict(X_input)
    lstm_closing_price = scaler.inverse_transform(lstm_closing_price)

    valid = new_dataset[160:]

    valid['LSTMPredictions'] = lstm_closing_price
    valid['RNNPrediction'] = rnn_closing_price
    valid['TransformerPrediction'] = transform_closing_price
    xCol = valid.index

    return {
        "data": [
            go.Scatter(
                name='Actual',
                x=xCol,
                y=valid["Close"],
                mode='lines'
            ),
            go.Scatter(
                name='LSTMPredictions',
                x=xCol,
                y=valid["LSTMPredictions"],
                mode='lines'
            ),
            go.Scatter(
                name='RNNPrediction',
                x=xCol,
                y=valid["RNNPrediction"],
                mode='lines'
            ),
            go.Scatter(
                name='TransformerPrediction',
                x=xCol,
                y=valid["TransformerPrediction"],
                mode='lines'
            )
        ],
        "layout": go.Layout(
            title='scatter plot',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'}
        )
    }


#
# @app.callback(Output('highlow', 'figure'), [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown):
#     dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
#     trace1 = []
#     trace2 = []
#     for stock in selected_dropdown:
#         trace1.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["High"],
#                        mode='lines', opacity=0.7,
#                        name=f'High {dropdown[stock]}', textposition='bottom center'))
#         trace2.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["Low"],
#                        mode='lines', opacity=0.6,
#                        name=f'Low {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1, trace2]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
#                                             '#FF7400', '#FFF400', '#FF0056'],
#                                   height=600,
#                                   title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
#                                   xaxis={"title": "Date",
#                                          'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'count': 6, 'label': '6M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'step': 'all'}])},
#                                          'rangeslider': {'visible': True}, 'type': 'date'},
#                                   yaxis={"title": "Price (USD)"})}
#     return figure
#
#
# @app.callback(Output('volume', 'figure'), [Input('my-dropdown2', 'value')])
# def update_graph(selected_dropdown_value):
#     dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
#     trace1 = []
#     for stock in selected_dropdown_value:
#         trace1.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["Volume"],
#                        mode='lines', opacity=0.7,
#                        name=f'Volume {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
#                                             '#FF7400', '#FFF400', '#FF0056'],
#                                   height=600,
#                                   title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
#                                   xaxis={"title": "Date",
#                                          'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'count': 6, 'label': '6M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'step': 'all'}])},
#                                          'rangeslider': {'visible': True}, 'type': 'date'},
#                                   yaxis={"title": "Transactions Volume"})}
#     return figure


if __name__ == '__main__':
    app.run_server(debug=True)
