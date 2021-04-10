import cvxpy as cvx
import numpy as np 

class RobustOptimiser():
    def __init__(self, mu, sigma, omega, w_old, psi, kappa, lambda_):
        self.w = self.calculate_weights(mu, sigma, omega, w_old, psi, kappa, lambda_)

    def calculate_weights(self, mu, sigma, omega, w_old, psi, kappa, lambda_):
        dim = mu.shape[0]
        w_rob = cvx.Variable(dim)
        Q = np.linalg.cholesky(omega)
        error_risk_term = cvx.norm(Q @ w_rob, 2)
        risk_aversion_term = cvx.quad_form(w_rob, sigma)
        obj = cvx.Maximize(w_rob @ mu - kappa * error_risk_term - lambda_/2 * risk_aversion_term)
        constr_1 = sum(w_rob)
        if len(w_old) == 0:
            obj = cvx.Maximize(w_rob @ mu - kappa * error_risk_term - lambda_/2 * risk_aversion_term)
            problem = cvx.Problem(obj, [constr_1 == 1])
        else:
            obj = cvx.Maximize(w_rob @ mu - kappa * error_risk_term - lambda_/2 * risk_aversion_term - psi * cvx.norm(w_rob - w_old, 1) ** 2)
            problem = cvx.Problem(obj, [constr_1 == 1])


        try:
            problem.solve(verbose = True)
        except:
            problem.solve(solver = 'SCS')
        if problem.value != np.inf:
            return w_rob.value
        else:
            print('The problem is unfeasible')
            return None

    def get_w_robust(self):
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
rebalancing_freq = 4*20
data = Dataloader(period, tickers, rebalancing_freq)
dates, tickers_close_info = data.get_close()
close_returns = pd.DataFrame()
i = 0
sharpe = []
for k in np.linspace(0.5, 9.5, 10):
    sharpe_l = []
    for l in np.linspace(1,10,10):
        for close_df in tickers_close_info:

            tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
            exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_mu()
            sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()
            omega = np.diag(np.diag(sigma))

            if i > 0:

                robust_0_df = (tickers_close_returns.multiply(robust_0_weights)).sum(axis = 1)
                tickers_close_returns['Portfolio Robust'] = robust_0_df

                tickers_close_returns.dropna(axis = 0, inplace = True)
                close_returns = close_returns.append(tickers_close_returns)

            else:
                i = 1
            
            robust_0 = RobustOptimiser(exp_returns, sigma, omega, k, l)
            robust_0_weights = robust_0.get_w_robust()
            robust_0_weights = robust_0_weights/sum(abs(robust_0_weights))
        sharpe_l.append(np.mean(close_returns['Portfolio Robust'])/np.std(close_returns['Portfolio Robust']))
    sharpe.append(sharpe_l)      

heatmap = pd.DataFrame(sharpe)  
heatmap.columns = np.linspace(1, 10, 9)
heatmap.index =  np.linspace(0.5, 9.5, 9)

import seaborn as sns
ax = sns.heatmap(heatmap)
plt.ylabel(r'$\kappa$')
plt.xlabel(r'$\lambda$')
plt.suptitle('Sharpe Ratio as a function of hyperparameters')
plt.show()

# close_returns = (close_returns + 1).cumprod(axis = 0)
# close_returns[['Portfolio Robust']].plot()
# # close_returns.plot()
# for date in dates:
#     plt.axvline(date)
# plt.ylabel('Cummulative Return')
# plt.suptitle('Return of Portfolio Compared')
# plt.legend()
# plt.show()
'''
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
for close_df in tickers_close_info:

    tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
    exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_mu()
    sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()
    omega = np.diag(np.diag(sigma))

    if i > 0:

        robust_0_df = (tickers_close_returns.multiply(robust_0_weights)).sum(axis = 1)
        robust_1_df = (tickers_close_returns.multiply(robust_1_weights)).sum(axis = 1)
        robust_2_df = (tickers_close_returns.multiply(robust_2_weights)).sum(axis = 1)

        tickers_close_returns['Portfolio Robust kappa = 0'] = robust_0_df
        tickers_close_returns['Portfolio Robust Omega'] = robust_1_df
        tickers_close_returns['Portfolio Robust Sigma'] = robust_2_df

        tickers_close_returns.dropna(axis = 0, inplace = True)
        close_returns = close_returns.append(tickers_close_returns)

    else:
        i = 1
    
    robust_0 = RobustOptimiser(exp_returns, sigma, omega,[], 0, 0, 8)
    robust_0_weights = robust_0.get_w_robust()
    robust_0_weights = robust_0_weights/sum(abs(robust_0_weights))

    robust_1 = RobustOptimiser(exp_returns, sigma, sigma,[], 0, 5, 8)
    robust_1_weights = robust_1.get_w_robust()
    robust_1_weights = robust_1_weights/sum(abs(robust_1_weights))

    robust_2 = RobustOptimiser(exp_returns, sigma, omega,[], 0, 5, 8)
    robust_2_weights = robust_2.get_w_robust()
    robust_2_weights = robust_2_weights/sum(abs(robust_2_weights))

v = {}
s = {}
portfolios = ['Portfolio Robust kappa = 0', 'Portfolio Robust Omega', 'Portfolio Robust Sigma']
for p in portfolios:
    v[p] = np.mean(close_returns[p]) - 1.65 * np.sqrt(21) * np.std(close_returns[p])
    s[p] = np.mean(close_returns[p])/np.std(close_returns[p])

df = pd.DataFrame([v, s], index = ['VaR 95 at 1mo horizon', 'Sharpe Ratio']).T
print(df.to_latex())
'''