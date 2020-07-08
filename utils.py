import numpy as np
from alpha_vantage.timeseries import TimeSeries


def get_data(your_api_key, stock_name):
    ts = TimeSeries(key=your_api_key, output_format='pandas')
    data, meta_data = ts.get_daily(stock_name, outputsize='full')
    return data


def slicing_50(x, history_points):
    sliced_data = np.array([x[i: i + history_points] for i in range(len(x) - history_points)])
    labels = np.array([x[:, 0][i + history_points] for i in range(len(x) - history_points)])
    return sliced_data, labels
