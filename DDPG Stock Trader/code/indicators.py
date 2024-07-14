import numpy as np

def ema(x, period=12):
    alpha = 2/(period+1)
    ema_series = []
    ema = x[0]
    ema_series.append(ema)
    for i in range(1, x.shape[0]):
        ema = ema*(1-alpha) + alpha*x[i]
        ema_series.append(ema)
    return np.array(ema_series)

def macd(x):
    return ema(x, period=12) - ema(x,period=26)

def rsi(open, close, period=14):
    rsi_series = []
    gain_series = []
    loss_series = []
    difference = close[0]-open[0]

    for i in range(0, min(period-1, open.shape[0])):
        difference = close[i]-open[i]
        if difference > 0:
            gain = difference
            loss = 1e-7
        elif difference < 0:
            loss = -difference
            gain = 1e-7
        rsi = 100 - 100/(1+gain/loss)
        rsi_series.append(rsi)
        gain_series.append(gain)
        loss_series.append(loss)

    avg_gain = np.mean(gain_series)
    avg_loss = np.mean(loss_series)

    for i in range(period-1, open.shape[0]):
        difference = close[i]/open[i] - 1
        if difference > 0:
            gain = difference
            loss = 1e-7
        elif difference < 0:
            loss = -difference
            gain = 1e-7
        rsi = 100 - 100/(1+(avg_gain*(period-1)+gain)/(avg_loss*(period-1)+loss))
        rsi_series.append(rsi)
        gain_series.append(gain)
        loss_series.append(loss)
        avg_gain = np.mean(gain_series[-period+1:])
        avg_loss = np.mean(loss_series[-period+1:])
    return np.array(rsi_series)