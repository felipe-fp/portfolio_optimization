from mu import *
from sigma import *
from dataloader import *
from markowitz import *
from robust_optimiser import *
from mean_variance_portfolio import *
from risk_portfolio import *
from black_litterman import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_dataprojets = pd.read_excel('portfolio_optimization/DataProjets.xls', sheet_name=0) 
df_marketcap = pd.read_excel('portfolio_optimization/DataProjets.xls', sheet_name=1, index_col=0)

correspondences = df_dataprojets[['Sedol', 'Tickers']]
correspondences.set_index('Sedol', drop = True, inplace = True)
correspondences_dict = correspondences.to_dict()['Tickers']

df_marketcap.index = pd.to_datetime(df_marketcap.index)
df_marketcap_month = df_marketcap.resample('1D')
df_marketcap_month = df_marketcap_month.mean()
df_marketcap_month.fillna(method = 'ffill', inplace = True)
df_marketcap_month.rename(columns = correspondences_dict, inplace = True)

tickers_list = list(df_dataprojets['Tickers'].values)
market_cap_tickers = set(df_marketcap_month.columns)
tickers_list_filtered = np.random.choice(tickers_list, size = 25)
tickers = list(market_cap_tickers.intersection(set(tickers_list_filtered)))
period = '1y'
rebalancing_freq = 2*21

data = Dataloader(period, tickers, rebalancing_freq)
dates, tickers_close_info = data.get_close()
tickers_marketcap = df_marketcap_month[tickers]

# Initialization:
close_returns = pd.DataFrame()
i = 0
prior = []
markowitz_weights = []
robust_weights = []
mvo_weights = []
psi = 0.3

for close_df in tickers_close_info:
    tickers_close_returns = (close_df/close_df.shift(1)).dropna() - 1
    exp_returns = mu(tickers_close_returns, tickers, rebalancing_freq).get_mu()
    sigma = Sigma(tickers_close_returns,rebalancing_freq).get_sigma()
    omega = np.diag(np.diag(sigma))

    if i > 0:

        markowitz_df = (tickers_close_returns.multiply(markowitz_weights)).sum(axis = 1)
        robust_df = (tickers_close_returns.multiply(robust_weights)).sum(axis = 1)
        mvo_df = (tickers_close_returns.multiply(mvo_weights)).sum(axis = 1)
        risk_inv_df = (tickers_close_returns.multiply(risk_weights_inv)).sum(axis = 1)
        risk_min_df = (tickers_close_returns.multiply(risk_weights_min)).sum(axis = 1)
        risk_eq_df = (tickers_close_returns.multiply(risk_weights_eq)).sum(axis = 1)
        black_litterman_df = (tickers_close_returns.multiply(bl_weights)).sum(axis = 1)

        tickers_close_returns['Portfolio Markowitz'] = markowitz_df
        tickers_close_returns['Portfolio Robust'] = robust_df
        tickers_close_returns['Portfolio MVO'] = mvo_df
        tickers_close_returns['Portfolio Risk Equal Weights'] = risk_eq_df
        tickers_close_returns['Portfolio Risk Min Variance'] = risk_min_df
        tickers_close_returns['Portfolio Risk Inv Variance'] = risk_inv_df
        tickers_close_returns['Portfolio Black Litterman'] = black_litterman_df

        tickers_close_returns.dropna(axis = 0, inplace = True)
        close_returns = close_returns.append(tickers_close_returns)

    else:
        i += 1
        end_date = list(tickers_close_returns.index)[-1]
        marketcap = tickers_marketcap.loc[end_date].values
    
    markowitz = Markowitz(exp_returns, sigma, markowitz_weights, psi, 3)
    markowitz_weights = markowitz.get_weights()
    markowitz_weights = markowitz_weights/sum(abs(markowitz_weights))

    robust = RobustOptimiser(exp_returns, sigma, omega, robust_weights, psi, 5, 8)
    robust_weights = robust.get_w_robust()
    robust_weights = robust_weights/sum(abs(robust_weights))

    mvo = MVO(exp_returns, sigma, mvo_weights, psi, 0.6)
    mvo_weights = mvo.get_weights()
    mvo_weights = mvo_weights/sum(abs(mvo_weights))

    risk_portfolio = RiskPortfolio(sigma)
    risk_weights_inv = np.transpose(risk_portfolio.get_weights('Inv Variance'))
    risk_weights_min = np.transpose(risk_portfolio.get_weights('Min Variance'))
    risk_weights_eq = np.transpose(risk_portfolio.get_weights('Equally Weighted'))

    black_litterman = BlackLittermanPortfolio(exp_returns, sigma, marketcap, prior, 0.1, 0.75, 0.2, omega)
    bl_weights = black_litterman.get_weights()
    bl_weights = bl_weights/sum(abs(bl_weights))
    prior = black_litterman.get_posterior()

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

# --------------------------------------------
# Sharpe Ratio Calculation
close_returns['Portfolio Markowitz'] = close_returns['Portfolio Markowitz'].astype(float)
close_returns['Portfolio Robust'] = close_returns['Portfolio Robust'].astype(float)
close_returns['Portfolio MVO'] = close_returns['Portfolio MVO'].astype(float)
close_returns['Portfolio Risk Equal Weights'] = close_returns['Portfolio Risk Equal Weights'].astype(float)
close_returns['Portfolio Risk Min Variance'] = close_returns['Portfolio Risk Min Variance'].astype(float)
close_returns['Portfolio Risk Inv Variance'] = close_returns['Portfolio Risk Inv Variance'].astype(float)

risk_free_return = 0.0
sharpe = {}
sharpe['Portfolio Markowitz'] = np.mean(close_returns['Portfolio Markowitz'] - risk_free_return)/np.std(close_returns['Portfolio Markowitz'])
sharpe['Portfolio Robust'] = np.mean(close_returns['Portfolio Robust'] - risk_free_return)/np.std(close_returns['Portfolio Robust'])
sharpe['Portfolio MVO'] = np.mean(close_returns['Portfolio MVO'] - risk_free_return)/np.std(close_returns['Portfolio MVO'])
sharpe['Portfolio Risk Equal Weights'] = np.mean(close_returns['Portfolio Risk Equal Weights'] - risk_free_return)/np.std(close_returns['Portfolio Risk Equal Weights'])
sharpe['Portfolio Risk Min Variance'] = np.mean(close_returns['Portfolio Risk Min Variance'] - risk_free_return)/np.std(close_returns['Portfolio Risk Min Variance'])
sharpe['Portfolio Risk Inv Variance'] = np.mean(close_returns['Portfolio Risk Inv Variance'] - risk_free_return)/np.std(close_returns['Portfolio Risk Inv Variance'])
sharpe['Portfolio Black Litterman'] = np.mean(close_returns['Portfolio Black Litterman'] - risk_free_return)/np.std(close_returns['Portfolio Black Litterman'])
df_sharpe = pd.DataFrame(sharpe, index = [0]).T
df_sharpe.columns = ['Sharpe Ratio']
print(df_sharpe.to_latex())

# --------------------------------------------------------
# Value at Risk Calculation
# 1 month horizon at 95%
VaR_1mo_95 = {}
VaR_1mo_95['Portfolio Markowitz'] = np.mean(close_returns['Portfolio Markowitz']) - 1.65 * np.sqrt(21) * np.std(close_returns['Portfolio Markowitz'])
VaR_1mo_95['Portfolio Robust'] = np.mean(close_returns['Portfolio Robust']) - 1.65 * np.sqrt(21) * np.std(close_returns['Portfolio Robust'])
VaR_1mo_95['Portfolio MVO'] = np.mean(close_returns['Portfolio MVO']) - 1.65 * np.sqrt(21) * np.std(close_returns['Portfolio MVO'])
VaR_1mo_95['Portfolio Risk Equal Weights'] = np.mean(close_returns['Portfolio Risk Equal Weights']) - 1.65 * np.sqrt(21) * np.std(close_returns['Portfolio Risk Equal Weights'])
VaR_1mo_95['Portfolio Risk Min Variance'] = np.mean(close_returns['Portfolio Risk Min Variance']) - 1.65 * np.sqrt(21) * np.std(close_returns['Portfolio Risk Min Variance'])
VaR_1mo_95['Portfolio Risk Inv Variance'] = np.mean(close_returns['Portfolio Risk Inv Variance']) - 1.65 * np.sqrt(21) * np.std(close_returns['Portfolio Risk Inv Variance'])
VaR_1mo_95['Portfolio Black Litterman'] = np.mean(close_returns['Portfolio Black Litterman']) - 1.65 * np.sqrt(21) * np.std(close_returns['Portfolio Black Litterman'])

# 1 month horizon at 99%
VaR_1mo_99 = {}
VaR_1mo_99['Portfolio Markowitz'] = np.mean(close_returns['Portfolio Markowitz']) - 2.33 * np.sqrt(21) * np.std(close_returns['Portfolio Markowitz'])
VaR_1mo_99['Portfolio Robust'] = np.mean(close_returns['Portfolio Robust']) - 2.33 * np.sqrt(21) * np.std(close_returns['Portfolio Robust'])
VaR_1mo_99['Portfolio MVO'] = np.mean(close_returns['Portfolio MVO']) - 2.33 * np.sqrt(21) * np.std(close_returns['Portfolio MVO'])
VaR_1mo_99['Portfolio Risk Equal Weights'] = np.mean(close_returns['Portfolio Risk Equal Weights']) - 2.33 * np.sqrt(21) * np.std(close_returns['Portfolio Risk Equal Weights'])
VaR_1mo_99['Portfolio Risk Min Variance'] = np.mean(close_returns['Portfolio Risk Min Variance']) - 2.33 * np.sqrt(21) * np.std(close_returns['Portfolio Risk Min Variance'])
VaR_1mo_99['Portfolio Risk Inv Variance'] = np.mean(close_returns['Portfolio Risk Inv Variance']) - 2.33 * np.sqrt(21) * np.std(close_returns['Portfolio Risk Inv Variance'])
VaR_1mo_99['Portfolio Black Litterman'] = np.mean(close_returns['Portfolio Black Litterman']) - 2.33 * np.sqrt(21) * np.std(close_returns['Portfolio Black Litterman'])

df_var = pd.DataFrame([VaR_1mo_95, VaR_1mo_99], index = ['VaR 95 at 1mo horizon', 'VaR 99 at 1mo horizon']).T
print(df_var.to_latex())

# -----------------------------------------
# Maximum Drawdown
# https://quant.stackexchange.com/questions/18094/how-can-i-calculate-the-maximum-drawdown-mdd-in-python
window = 21
returns = close_returns[['Portfolio Markowitz', 'Portfolio Robust', 'Portfolio MVO', 'Portfolio Risk Equal Weights', 'Portfolio Risk Min Variance', 'Portfolio Risk Inv Variance', 'Portfolio Black Litterman']]
Roll_Max = returns.rolling(window, min_periods=1).max()
Daily_Drawdown = returns/Roll_Max - 1.0
Max_Daily_Drawdown = Daily_Drawdown.min(axis = 0)
Max_Daily_Drawdown.columns = ['Maximum 1 week Drawdown']
print(Max_Daily_Drawdown.to_latex())


# ------------------------------------------
# Plotting Results
close_returns = (close_returns + 1).cumprod(axis = 0)
close_returns[['Portfolio Markowitz', 'Portfolio Robust', 'Portfolio MVO', 'Portfolio Risk Equal Weights', 'Portfolio Risk Min Variance', 'Portfolio Risk Inv Variance', 'Portfolio Black Litterman']].plot()
for date in dates:
    plt.axvline(date)
plt.ylabel('Cummulative Return')
plt.suptitle('Return of Portfolio vs. Individual Stocks')
plt.show()