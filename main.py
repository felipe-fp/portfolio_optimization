from mu import *
from sigma import *
from dataloader import *
from markowitz import *

tickers = ['AAPL', 'GOOG', 'TSLA', 'GME']
period = '12mo'
data = Dataloader('12mo', tickers)
tickers_close_price = data.get_close()
tickers_close_returns = (tickers_close_price/tickers_close_price.shift(1)).dropna() - 1
exp_returns = mu(tickers_close_returns).get_expected_returns()
sigma = Sigma(tickers_close_returns, 1).get_sigma()
m = Markowitz(exp_returns, sigma, 2)
print('Expected Returns Vector: \n')
print(exp_returns)
print('\n ------------- \n')

print('Covariance Matrix: \n')
print(sigma)
print('\n ------------- \n ')

print('Optima Weights: \n')
print(m.get_weights())
print('\n ------------- \n ')