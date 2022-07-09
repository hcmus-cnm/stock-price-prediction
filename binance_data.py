
from binance.client import Client
client = Client('s82OPcyd6IsqQyzckYbUa3GqzSUy27Gzljxn3YUaX6YqZ5txyYKdCac535O9E9at',
                '8cUcebJYub1eby34VfHAZEvePpeHV9osr78KQGJhXBpK20LBc8zO35TpPTnMgd54')


def get_binance_data(full=False):
    trades = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
    batch = []
    for trade in trades:
        (open_time, open_price, high_price, low_price, close_price, vol, close_time, *_) = trade
        if full:
            batch.append((open_time, open_price,  high_price, low_price, close_price, vol))
        else:
            batch.append((open_time, open_price))
    return batch



