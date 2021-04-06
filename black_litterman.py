import numpy as np
import cvxpy as cvx

class BlackLittermanPortfolio():
    def __init__(self, exp_returns, sigma, market_cap, prior, delta, tau, min_returns, omega):
        if len(prior) == 0:
            self.prior = self.calculate_prior(sigma, market_cap, delta, tau)
        else:
            self.prior = prior
        self.likelihood = self.calculate_likelihood(exp_returns, sigma, tau)
        self.posterior = self.master_formula(self.prior, self.likelihood)
        self.optimal_weights = self.optimize_problem(self.posterior, omega, min_returns)

    def calculate_prior(self, sigma, market_cap, delta, tau):
        market_cap = (np.array(market_cap)).reshape((-1,1))
        equilibrium_returns = delta * (sigma @ market_cap)
        return [equilibrium_returns, tau * sigma]
    
    def calculate_likelihood(self, expected_returns, sigma, tau):
        views_vector = (np.array(expected_returns)).reshape((-1,1))
        absolute_pick_matrix = np.identity(views_vector.shape[0])
        omega = np.diag(np.diag(tau * absolute_pick_matrix @ sigma @ absolute_pick_matrix))
        return [views_vector, omega]

    def master_formula(self, prior, likelihood):
        mu_prior, sigma_prior = prior[0], prior[1]
        mu_likelihood, sigma_likelihood = likelihood[0], likelihood[1]

        mu_bl = np.linalg.inv(np.linalg.inv(sigma_prior) + np.linalg.inv(sigma_likelihood)) @\
            (np.linalg.inv(sigma_prior)@ mu_prior + np.linalg.inv(sigma_likelihood) @ mu_likelihood)
        sigma_bl = np.linalg.inv(np.linalg.inv(sigma_prior) + np.linalg.inv(sigma_likelihood))
        
        return [mu_bl, sigma_bl]

    def optimize_problem(self, posterior, omega, min_returns):
        mu, sigma = posterior[0], posterior[1]

        dim = mu.shape[0]
        w = cvx.Variable(dim)
        obj = cvx.Minimize(cvx.quad_form(w, sigma))
        mu_t = np.reshape(mu, (1, -1))
        constr_1 = mu_t @ w 
        constr_2 = sum(w)
        problem = cvx.Problem(obj, [constr_1 >= min_returns, constr_2 == 1])
        problem.solve()

        return w.value
    
    def get_weights(self):
        return self.optimal_weights
    def get_posterior(self):
        return self.posterior

'''
import pandas as pd 
df_dataprojets = pd.read_excel('portfolio_optimization/DataProjets.xls', sheet_name=0) 
df_marketcap = pd.read_excel('portfolio_optimization/DataProjets.xls', sheet_name=1, index_col=0)
df_sector = pd.read_excel('portfolio_optimization/DataProjets.xls', sheet_name=2) 
df_bechmark =pd.read_excel('portfolio_optimization/DataProjets.xls', sheet_name=3, index_col = 0) 

tickers = list(df_dataprojets['Tickers'].values)

df_marketcap.index = pd.to_datetime(df_marketcap.index)
df_marketcap_month = df_marketcap.resample('1D')
df_marketcap_month = df_marketcap_month.mean()
df_marketcap_month.fillna(method = 'ffill', inplace = True)

correspondences = df_dataprojets[['Sedol', 'Tickers']]
correspondences.set_index('Sedol', drop = True, inplace = True)
correspondences_dict = correspondences.to_dict()['Tickers']
df_marketcap_month.rename(columns = correspondences_dict, inplace = True)

tickers = ['AAPL', 'IBM', 'BLK', 'AMZN', 'XOM', 'NKE']
tickers_marketcap = df_marketcap_month[tickers]

from mu import *
from sigma import *
from dataloader import *
from markowitz import *
from robust_optimiser import *
from mean_variance_portfolio import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

period = '12mo'
rebalancing_freq = 5*20
data = Dataloader(period, tickers, rebalancing_freq)
dates, tickers_close_info = data.get_close()
close_returns = pd.DataFrame()
i = 0
prior = []
for close_df in tickers_close_info:

    tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
    exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_mu()
    # exp_returns = (tickers_close_returns.iloc[-1].values)
    sigma = Sigma(tickers_close_returns, rebalancing_freq).get_sigma()
    # sigma = tickers_close_returns.cov().to_numpy()
    omega=np.diag(np.diag(sigma))
    if i > 0:
        markowitz_df = (tickers_close_returns.multiply(markowitz_weights)).sum(axis = 1)
        mvo_df = (tickers_close_returns.multiply(mvo_weights)).sum(axis = 1)

        tickers_close_returns['Portfolio Black-Litterman'] = markowitz_df
        tickers_close_returns['Portfolio Robust'] = mvo_df

        tickers_close_returns.dropna(axis = 0, inplace = True)
        close_returns = close_returns.append(tickers_close_returns)

    else:
        i = 1
        end_date = list(tickers_close_returns.index)[-1]
        marketcap = tickers_marketcap.loc[end_date].values

    markowitz = BlackLittermanPortfolio(exp_returns, sigma, marketcap, prior, 0.1, 0.75, 0.2, omega)
    mvo = RobustOptimiser(exp_returns, sigma, omega, 5,8)
    markowitz_weights = markowitz.get_weights()
    mvo_weights = mvo.get_w_robust()
    markowitz_weights = markowitz_weights/sum(abs(markowitz_weights))
    mvo_weights = mvo_weights/sum(abs(mvo_weights))
    prior = markowitz.get_posterior()
    
close_returns = (close_returns + 1).cumprod(axis = 0)
close_returns[['Portfolio Black-Litterman', 'Portfolio Robust']].plot()
plt.show()
'''

    
