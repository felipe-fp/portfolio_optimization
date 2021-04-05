import numpy as np
import pandas as pd
import cvxpy as cvx

class Markowitz():
    def __init__(self, mu, sigma, max_var):
        self.w = self.calculate_weights(mu, sigma, max_var)

    def calculate_weights(self, mu, sigma, max_var):
        dim = mu.shape[0]
        w_opt = cvx.Variable(dim)
        obj = cvx.Maximize(w_opt @ mu)
        contr_1 = cvx.quad_form(w_opt, sigma)
        constr_2 = sum(w_opt)
        # problem = cvx.Problem(obj, [contr_1 <= max_var, constr_2 == 1])
        problem = cvx.Problem(obj, [contr_1 <= max_var])
        try:
            problem.solve(verbose = True)
        except:
            problem.solve(solver = 'SCS')
        w = w_opt.value

        return w
    def get_weights(self):
        return self.w


from mu import *
from sigma import *
from dataloader import *
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
    sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()

    if i > 0:

        markowitz_df = (tickers_close_returns.multiply(markowitz_weights)).sum(axis = 1)
        tickers_close_returns['Portfolio Markowitz'] = markowitz_df
        tickers_close_returns.dropna(axis = 0, inplace = True)
        close_returns = close_returns.append(tickers_close_returns)

    else:
        i = 1
    
    markowitz = Markowitz(exp_returns, sigma, 3.8)
    markowitz_weights = markowitz.get_weights()
    markowitz_weights = markowitz_weights/sum(abs(markowitz_weights))


close_returns = (close_returns + 1).cumprod(axis = 0)
close_returns[['Portfolio Markowitz', 'Portfolio Robust', 'Portfolio MVO']].plot()
# close_returns.plot()
for date in dates:
    plt.axvline(date)
plt.ylabel('Cummulative Return')
plt.suptitle('Return of Portfolio vs. Individual Stocks')
plt.show()