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

tickers = ['AAPL', 'GOOG', 'IBM', 'TSLA', 'BLK', 'AMZN', 'COTY', 'PFE']
period = '12mo'
rebalancing_freq = 5*20
data = Dataloader(period, tickers, rebalancing_freq)
dates, tickers_close_info = data.get_close()
close_returns = pd.DataFrame()
i = 0
s = []
a = np.linspace(0.01, 1, 5)
for m in a:
    for close_df in tickers_close_info:

        tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
        exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_mu()

        sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()

        if i > 0:

            mvo1_df = (tickers_close_returns.multiply(mvo1_weights)).sum(axis = 1)

            tickers_close_returns['Portfolio Min Variance'] = mvo1_df

            tickers_close_returns.dropna(axis = 0, inplace = True)
            close_returns = close_returns.append(tickers_close_returns)

        else:
            i = 1
        
        mvo1 = MVO(exp_returns, sigma, m)
        mvo1_weights = mvo1.get_weights()
        mvo1_weights = mvo1_weights/sum(abs(mvo1_weights))

    # close_returns = (close_returns + 1).cumprod(axis = 0)
    s.append(np.mean(close_returns['Portfolio Min Variance'])/np.std(close_returns['Portfolio Min Variance']))
plt.plot(a, s)
plt.xlabel(r'$\mu_{min}$')
plt.ylabel('Sharpe Ratio') 
plt.suptitle('Minimum Variance Portfolio Performance')
plt.show()
# close_returns[['Portfolio Min Variance']].plot()
# for date in dates:
#     plt.axvline(date)
# plt.ylabel('Cummulative Return')
# plt.suptitle(r'Return of Portfolio for different $\mu_{min}}$')
# plt.show()
'''