import numpy as np
import pandas as pd
import cvxpy as cvx

class Markowitz():
    def __init__(self, mu, sigma, w_old, psi, max_var):
        self.w = self.calculate_weights(mu, sigma, w_old, psi, max_var)

    def calculate_weights(self, mu, sigma, w_old, psi, max_var):
        dim = mu.shape[0]
        w_opt = cvx.Variable(dim)
        contr_1 = cvx.quad_form(w_opt, sigma)
        constr_2 = sum(w_opt)
        if len(w_old) == 0:
            obj = cvx.Maximize(w_opt @ mu)
            problem = cvx.Problem(obj, [contr_1 <= max_var, constr_2 == 1])
        else:
            obj = cvx.Maximize(w_opt @ mu - psi * cvx.norm(w_opt - w_old, 1)**2)
            problem = cvx.Problem(obj, [contr_1 <= max_var, constr_2 == 1])

        try:
            problem.solve(verbose = True)
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
rebalancing_freq = 4*21
data = Dataloader(period, tickers, rebalancing_freq)
dates, tickers_close_info = data.get_close()
close_returns = pd.DataFrame()
i = 0
for close_df in tickers_close_info:

    tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
    exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_expected_returns()
    sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()

    if i > 0:

        markowitz_1_df = (tickers_close_returns.multiply(markowitz_1_weights)).sum(axis = 1)
        markowitz_2_df = (tickers_close_returns.multiply(markowitz_2_weights)).sum(axis = 1)
        markowitz_3_df = (tickers_close_returns.multiply(markowitz_3_weights)).sum(axis = 1)

        tickers_close_returns['Portfolio Markowitz sigma = 1'] = markowitz_1_df
        tickers_close_returns['Portfolio Markowitz sigma = 2'] = markowitz_2_df
        tickers_close_returns['Portfolio Markowitz sigma = 0.5'] = markowitz_3_df

        tickers_close_returns.dropna(axis = 0, inplace = True)
        close_returns = close_returns.append(tickers_close_returns)

    else:
        i = 1
    
    markowitz1 = Markowitz(exp_returns, sigma, 1)
    markowitz2 = Markowitz(exp_returns, sigma, 2)
    markowitz3 = Markowitz(exp_returns, sigma, 0.5)

    markowitz_1_weights = markowitz1.get_weights()
    markowitz_2_weights = markowitz2.get_weights()
    markowitz_3_weights = markowitz3.get_weights()

close_returns = (close_returns + 1).cumprod(axis = 0)
close_returns[['Portfolio Markowitz sigma = 0.5', 'Portfolio Markowitz sigma = 2', 'Portfolio Markowitz sigma = 1']].plot()
# close_returns.plot()
for date in dates:
    plt.axvline(date)
plt.ylabel('Cummulative Return')
plt.suptitle('Return of Portfolio')
plt.show()

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
a = np.linspace(0.5, 4, 5)
for m in a:
    for close_df in tickers_close_info:

        tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
        exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_mu()

        sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()

        if i > 0:

            mvo1_df = (tickers_close_returns.multiply(mvo1_weights)).sum(axis = 1)

            tickers_close_returns['Portfolio Max Returns'] = mvo1_df

            tickers_close_returns.dropna(axis = 0, inplace = True)
            close_returns = close_returns.append(tickers_close_returns)

        else:
            i = 1
        
        mvo1 = Markowitz(exp_returns, sigma, m)
        mvo1_weights = mvo1.get_weights()
        mvo1_weights = mvo1_weights/sum(abs(mvo1_weights))

    # close_returns = (close_returns + 1).cumprod(axis = 0)
    s.append(np.mean(close_returns['Portfolio Max Returns'])/np.std(close_returns['Portfolio Max Returns']))
plt.plot(a, s)
plt.xlabel(r'$\mu_{min}$')
plt.ylabel('Sharpe Ratio') 
plt.suptitle('Minimum Variance Portfolio Performance')
plt.show()
''' 