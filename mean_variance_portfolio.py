import cvxpy as cvx

class MVO():
    def __init__(self, mu, sigma, w_old, psi, min_returns):
        self.w = self.calculate_weights(mu, sigma, w_old, psi, min_returns)

    def calculate_weights(self, mu, sigma, w_old, psi, min_returns):
        dim = mu.shape[0]
        w_opt = cvx.Variable(dim)
        constr_1 = w_opt @ mu
        constr_2 = sum(w_opt)
        if len(w_old) == 0:
            obj = cvx.Minimize(cvx.quad_form(w_opt, sigma))
            problem = cvx.Problem(obj, [constr_1 >= min_returns, constr_2 == 1])
        else:
            obj = cvx.Minimize(cvx.quad_form(w_opt, sigma) + psi * cvx.norm(w_opt - w_old, 1)**2)
            problem = cvx.Problem(obj, [constr_1 >= min_returns, constr_2 == 1])

        try:
            problem.solve(verbose = False)
        except:
            problem.solve(solver = 'SCS')
        w = w_opt.value
        return w
    def get_weights(self):
        return self.w

'''
from mu import *
from sigma import *
from dataloader import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
psi = 0
tickers = ['AAPL', 'GOOG', 'IBM', 'TSLA', 'BLK', 'AMZN', 'COTY', 'PFE']
period = '12mo'
rebalancing_freq = 5*20
data = Dataloader(period, tickers, rebalancing_freq)
dates, tickers_close_info = data.get_close()
close_returns = pd.DataFrame()
i = 0
s = []
for close_df in tickers_close_info:

    tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
    exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_mu()

    sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()

    if i > 0:

        mvo1_df = (tickers_close_returns.multiply(mvo1_weights)).sum(axis = 1)
        mvo2_df = (tickers_close_returns.multiply(mvo2_weights)).sum(axis = 1)
        mvo3_df = (tickers_close_returns.multiply(mvo3_weights)).sum(axis = 1)

        tickers_close_returns['Portfolio Min Variance mu = 0.2'] = mvo1_df
        tickers_close_returns['Portfolio Min Variance mu = 0.5'] = mvo2_df
        tickers_close_returns['Portfolio Min Variance mu = 1'] = mvo3_df

        tickers_close_returns.dropna(axis = 0, inplace = True)
        close_returns = close_returns.append(tickers_close_returns)

    else:
        i = 1
    
    mvo1 = MVO(exp_returns, sigma,[],psi, 0.2)
    mvo2 = MVO(exp_returns, sigma,[],psi, 0.5)
    mvo3 = MVO(exp_returns, sigma,[],psi, 1)

    mvo1_weights = mvo1.get_weights()
    mvo1_weights = mvo1_weights/sum(abs(mvo1_weights))

    mvo2_weights = mvo2.get_weights()
    mvo2_weights = mvo2_weights/sum(abs(mvo2_weights)) 

    mvo3_weights = mvo3.get_weights()
    mvo3_weights = mvo3_weights/sum(abs(mvo3_weights))

close_returns = (close_returns + 1).cumprod(axis = 0)
close_returns[['Portfolio Min Variance mu = 0.2', 'Portfolio Min Variance mu = 0.5', 'Portfolio Min Variance mu = 1']].plot()
for date in dates:
    plt.axvline(date)
plt.ylabel('Cummulative Return')
plt.suptitle(r'Return of Portfolio for different $\mu_{min}}$')
plt.show()
 '''