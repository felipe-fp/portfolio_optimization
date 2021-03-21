from mu import *
from sigma import *
from dataloader import *
from markowitz import *
from robust_optimizer import *
import matplotlib.pyplot as plt
import pandas as pd

tickers = ['AAPL', 'GOOG', 'TSLA', 'BTC-USD']
period = '12mo'
data = Dataloader('24mo', tickers, 3)
tickers_close_price = data.get_close()
close_returns = pd.DataFrame()

for close_df in tickers_close_price:
    tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
    exp_returns = mu(tickers_close_returns).get_expected_returns()
    sigma = Sigma(tickers_close_returns, 90).get_sigma()
    markowitz = Markowitz(exp_returns, sigma, 2.2)

    print('Expected Returns Vector: \n')
    print(exp_returns)
    print('\n ------------- \n')

    print('Covariance Matrix: \n')
    print(sigma)
    print('\n ------------- \n ')

    print('Optima Weights: \n')
    print(markowitz.get_weights())
    print('\n')
    print('\n ------------- \n ')

    tickers_close_returns['Portfolio'] = (tickers_close_returns.multiply(markowitz.get_weights())).sum(axis = 1)
    tickers_close_returns.dropna(axis = 0, inplace = True)
    close_returns = close_returns.append(tickers_close_returns)

close_returns = (close_returns + 1).cumprod(axis = 0)
close_returns.plot()
plt.ylabel('Cummulative Return')
plt.suptitle('Return of Portfolio vs. Individual Stocks')
plt.show()