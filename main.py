from mu import *
from sigma import *
from dataloader import *
from markowitz import *
from robust_optimiser import *
from mean_variance_portfolio import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tickers = ['AAPL', 'GOOG', 'IBM', 'TSLA', 'BLK', 'AMZN', 'COTY', 'PFE']
period = '12mo'
rebalancing_freq = 20
data = Dataloader(period, tickers, rebalancing_freq)
dates, tickers_close_info = data.get_close()
close_returns = pd.DataFrame()
i = 0
for close_df in tickers_close_info:

    tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
    exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_mu()
    sigma = Sigma(tickers_close_returns, 30 * rebalancing_freq).get_sigma()
    omega = np.diag(np.diag(sigma))

    if i > 0:

        markowitz_df = (tickers_close_returns.multiply(markowitz_weights)).sum(axis = 1)
        robust_df = (tickers_close_returns.multiply(robust_weights)).sum(axis = 1)
        mvo_df = (tickers_close_returns.multiply(mvo_weights)).sum(axis = 1)

        tickers_close_returns['Portfolio Markowitz'] = markowitz_df
        tickers_close_returns['Portfolio Robust'] = robust_df
        tickers_close_returns['Portfolio MVO'] = mvo_df

        tickers_close_returns.dropna(axis = 0, inplace = True)
        close_returns = close_returns.append(tickers_close_returns)

    else:
        i = 1
    
    markowitz = Markowitz(exp_returns, sigma, 3.8)
    markowitz_weights = markowitz.get_weights()
    markowitz_weights = markowitz_weights/sum(abs(markowitz_weights))

    robust = RobustOptimiser(exp_returns, sigma, omega, 5, 8)
    robust_weights = robust.get_w_robust()
    robust_weights = robust_weights/sum(abs(robust_weights))

    mvo = MVO(exp_returns, sigma, 0.2)
    mvo_weights = mvo.get_weights()
    mvo_weights = mvo_weights/sum(abs(mvo_weights))

    print('Expected Returns Vector: \n')
    print(exp_returns)
    print('\n ------------- \n')

    print('Covariance Matrix: \n')
    print(sigma)
    print('\n ------------- \n ')

    print('Optimal Weights Markowitz: \n')
    print(markowitz_weights)
    print('\n')
    print('Optimal Weights Robust: \n')
    print(robust_weights)
    print('\n ------------- \n ')
    print('Optimal Weights MVO: \n')
    print(mvo_weights)
    print('\n ------------- \n ')

close_returns = (close_returns + 1).cumprod(axis = 0)
close_returns[['Portfolio Markowitz', 'Portfolio Robust', 'Portfolio MVO']].plot()
# close_returns.plot()
for date in dates:
    plt.axvline(date)
plt.ylabel('Cummulative Return')
plt.suptitle('Return of Portfolio vs. Individual Stocks')
plt.show()