import numpy as np
import pandas as pd

class RiskPortfolio():
    def __init__(self, sigma):
        dim = sigma.shape[0]
        self.w_equaly_weighted = self.equaly_weighted(dim)
        self.w_inv_variance = self.inv_variance(sigma, dim)
        self.w_min_variance = self.min_variance(sigma, dim)

    def equaly_weighted(self, dim):
        identity = np.identity(dim)
        ones = np.ones((dim, 1))
    
        return (identity @ ones)/(ones.T @ identity @ ones)

    def inv_variance(self, sigma, dim):
        lambda_ = np.diag(np.diag(sigma))
        lambda_2 = lambda_ ** 2
        ones = np.ones((dim, 1))

        return (np.linalg.inv(lambda_2) @ ones)/ (ones.T @ np.linalg.inv(lambda_2) @ ones)

    def min_variance(self, sigma, dim):
        ones = np.ones((dim, 1))

        return (np.linalg.inv(sigma) @ ones) / (ones.T @ np.linalg.inv(sigma) @ ones)
    
    def get_weights(self, type):
        if type == 'Min Variance':
            return self.w_min_variance
        elif type == 'Inv Variance':
            return self.w_inv_variance
        elif type == 'Equally Weighted':
            return self.w_equaly_weighted
        else:
            print('Not a Valid Type. You can try one of the following: Min Variance, Inv Variance, Equally Weighted')
            return None
'''
from mu import *
from sigma import *
from dataloader import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tickers = ['AAPL', 'GOOG', 'IBM', 'TSLA', 'BLK', 'AMZN', 'COTY', 'PFE']
period = '12mo'
rebalancing_freq = 5*21
data = Dataloader(period, tickers, rebalancing_freq)
dates, tickers_close_info = data.get_close()
close_returns = pd.DataFrame()
i = 0
s = []
for close_df in tickers_close_info:

    tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
    sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()

    if i > 0:

        risk_inv_df = (tickers_close_returns.multiply(risk__weights_inv)).sum(axis = 1)

        risk_min_df = (tickers_close_returns.multiply(risk__weights_min)).sum(axis = 1)

        risk_eq_df = (tickers_close_returns.multiply(risk__weights_eq)).sum(axis = 1)

        tickers_close_returns['Portfolio Risk Inverse Variance'] = risk_inv_df
        tickers_close_returns['Portfolio Risk Min Variance'] = risk_min_df
        tickers_close_returns['Portfolio Risk Equally Weighted'] = risk_eq_df

        tickers_close_returns.dropna(axis = 0, inplace = True)
        close_returns = close_returns.append(tickers_close_returns)

    else:
        i = 1
    
    risk_ = RiskPortfolio(sigma)
    risk__weights_inv = np.transpose(risk_.get_weights('Inv Variance'))[0]
    risk__weights_min = np.transpose(risk_.get_weights('Min Variance'))[0]
    risk__weights_eq = np.transpose(risk_.get_weights('Equally Weighted'))[0]

close_returns = (close_returns + 1).cumprod(axis = 0)
close_returns[['Portfolio Risk Inverse Variance', 'Portfolio Risk Min Variance', 'Portfolio Risk Equally Weighted']].plot()
plt.ylabel('Cummulative Returns') 
plt.suptitle('Minimum Variance Portfolio Performance')
plt.show()
'''