
import dash
from dash import dcc, Output, Input, html
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from xgboost_pred import Time2Vector, SingleAttention, MultiAttention, TransformerEncoder

app = dash.Dash()
server = app.server
window_size = 80

import pandas as pd
import numpy as np
from binance_data import get_binance_data

lstm_model = load_model("saved_lstm_model.h5")
xgboost_model = load_model("saved_rnn_model.h5")

transform_model = load_model('Transformer+TimeEmbedding.hdf5',
                             custom_objects={'Time2Vector': Time2Vector,
                                             'SingleAttention': SingleAttention,
                                             'MultiAttention': MultiAttention,
                                             'TransformerEncoder': TransformerEncoder})

# xgboost_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
# xgboost_model.load_model('xgboost.model')


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
    new_data1 = pd.DataFrame(get_binance_data(True), columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

    new_data1["Time"] = pd.to_datetime(new_data1['Time'], unit='ms')

    new_dataset = pd.DataFrame(index=range(0, len(new_data1)), columns=['Time', 'Close'])

    for i in range(0, len(new_data1)):
        new_dataset["Time"][i] = new_data1['time'][i]
        new_dataset["Close"][i] = new_data1["price"][i]
    # new_data = pd.DataFrame(get_binance_data(), columns=['Time', 'Close'])

    new_data1.index = new_data1.Time
    new_data1.drop("Time", axis=1, inplace=True)
    dataset = new_data1.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(new_dataset)

    inputs = new_dataset[window_size:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_input = np.array([
        inputs[i - window_size:i, 0]
        for i in range(window_size, inputs.shape[0])
    ])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    transform_closing_price = transform_model.predict(X_input)
    transform_closing_price = scaler.inverse_transform(transform_closing_price)

    xgboost_closing_price = xgboost_model.predict(X_input)
    xgboost_closing_price = scaler.inverse_transform(xgboost_closing_price)

    lstm_closing_price = lstm_model.predict(X_input)
    lstm_closing_price = scaler.inverse_transform(lstm_closing_price)

    valid = new_dataset[160:]

    valid['LSTMPredictions'] = lstm_closing_price
    valid['XGBoostPredictions'] = xgboost_closing_price
    valid['TransformPredictions'] = transform_closing_price
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
                name='XGBoostPredictions',
                x=xCol,
                y=valid["XGBoostPredictions"],
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
