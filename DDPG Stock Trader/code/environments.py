class Environment():
    def __init__(self):
        raise NotImplementedError
    def step(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

import yfinance as yf
import indicators
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Stock_Market(Environment):
    def __init__(self, starting_balance=10000):
        self.starting_balance=starting_balance
        self.balance = self.starting_balance
        self.prev_portfolio_value = self.starting_balance
        self.portfolio_value = self.starting_balance
        self.portfolio_values = []
        self.shares = 0
        self.indices = yf.Ticker('TSLA').history(start="2014-01-01", end="2020-01-01", interval="1d")[['Open', 'Close', 'High']]
        self.indices_min = dict(zip(self.indices.columns, self.indices.min(axis=0)))
        self.indices_max = dict(zip(self.indices.columns, self.indices.max(axis=0)))
        self.shares_max = self.starting_balance/self.indices['High'].max()
        # self.indices = self.indices.drop(['High'], axis=1)
        self.train_size = self.indices.shape[0]
        self.max_transaction = 10
        self.timestep = 0
        self.truncated = False
        self.penalty = 100 # in multiples of starting balance
        self.indices['EMA'] = indicators.ema(np.squeeze(self.indices[['Close']].to_numpy()))
        self.indices['MACD'] = indicators.macd(np.squeeze(self.indices[['Close']].to_numpy()))
        self.indices['RSI'] = indicators.rsi(np.squeeze(self.indices[['Open']].to_numpy()), np.squeeze(self.indices[['Close']].to_numpy()))
        # self.indices = self.indices.drop(['Open'], axis=1)
        for column in self.indices.columns:
            self.indices[column] = self.transform(self.indices[column])
        # axis = plt.subplot()
        # axis.plot(self.indices['Open'])
        # axis.plot(self.indices['Close'])
        # axis.plot(self.indices['High'])
        # axis.plot(self.indices['EMA'])
        # axis.plot(self.indices['RSI'])
        # plt.show()
    def step(self, action):
        current_price = self.indices['Close'].iloc[self.timestep%self.train_size]
        current_price = (self.indices_max['Close']-self.indices_min['Close'])*current_price + self.indices_min['Close']
        self.balance += -current_price*action[0]*self.max_transaction
        self.shares += action[0]*self.max_transaction

        self.portfolio_value = self.balance + self.shares*current_price
        self.portfolio_values.append(self.portfolio_value)
        reward = np.log(self.portfolio_value/self.prev_portfolio_value)
        self.prev_portfolio_value = self.portfolio_value
        # if self.balance < 0:
        #     reward -= self.penalty*self.starting_balance
        #     self.truncated = True
        # if self.shares < 0:
        #     reward -= self.penalty*self.starting_balance
        #     self.truncated = True
        next_state = np.append(self.indices.iloc[self.timestep%self.train_size].to_numpy(), [self.balance/self.starting_balance, self.shares/(self.shares_max)])

        self.timestep += 1
        if self.timestep>=self.indices.shape[0]:
            self.truncated = True
        return next_state, reward/self.starting_balance, None, self.truncated, None

    def reset(self):
        self.truncated = False
        self.balance = self.starting_balance
        self.shares = 0
        self.timestep = 0
        next_state = self.indices.iloc[self.timestep].to_numpy()
        next_state = np.append(next_state, [self.balance, self.shares])
        self.timestep += 1
        return next_state, None

    def transform(self, series: pd.Series):
        return (series - series.min())/(series.max() - series.min())
