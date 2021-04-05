from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from dataloader import *
import matplotlib.pyplot as plt
import sys
import yfinance as yf

sys.stderr = open('somefile', 'w')

class mu():
    def __init__(self, ticker_series, tickers, rebalancing_freq):
        aux = zip(ticker_series, tickers)
        self.expected_returns = np.array([self.fit_ARIMA(ticker_series[ticker_time_series], rebalancing_freq) for ticker_time_series in ticker_series])
        self.recommendations = np.array([self.fit_recommendations(ticker_series[ticker_time_series], ticker_name) for ticker_time_series, ticker_name in aux])
    def fit_ARIMA(self, time_series, rebalancing_freq):
        best_perf = np.inf
        for p_ in range(0,5):
            for d_ in range(0,3):
                for q_ in range(0,5):
                    model = ARIMA(endog=time_series, order = (p_,d_,q_), enforce_stationarity=True)
                    try:
                        model_fit = model.fit()
                        current_perf = model_fit.aic
                        if current_perf < best_perf:
                            mu = (np.prod(model_fit.forecast(rebalancing_freq).values + 1))
                            best_perf = current_perf
                    except:
                        pass
        return mu
    def fit_recommendations(self, time_series, ticker_name):
        print(ticker_name)
        start_date = time_series.index.values[0]
        end_date = time_series.index.values[-1]
        try:
            recommendations = yf.Ticker(ticker_name).recommendations.loc[start_date:end_date].groupby(['To Grade']).count()['Firm']
            recommendations_attr = list(recommendations.index)

            score = 0
            if 'Buy' in recommendations_attr:
                score += (1 * recommendations.loc['Buy'])
            if 'Sell' in recommendations_attr:
                score -= (1 * recommendations.loc['Sell'])
            if 'Strong Buy' in recommendations_attr:
                score += (2 * recommendations.loc['Strong Buy'])
            if 'Strong Sell' in recommendations_attr:
                score -= (2 * recommendations.loc['Strong Sell'])
            if 'Outperform' in recommendations_attr:
                score += (0.5 * recommendations.loc['Outperform'])
            if 'Underperform' in recommendations_attr:
                score -= (0.5* recommendations.loc['Underperform'])
            if len(recommendations_attr) > 0:
                return score/len(recommendations_attr)
            else:
                return 0
        except:
            return 0
    def get_expected_returns(self):
        return self.expected_returns
    def get_recommendations(self):
        return self.recommendations
    def get_mu(self):
        return (self.recommendations/sum(abs(self.recommendations)) + self.expected_returns/sum(abs(self.expected_returns)))


# tickers = ['IBM', 'BLK', 'AMZN', 'COTY', 'PFE']
# period = '12mo'
# data = Dataloader('12mo', tickers, 12)
# tickers_close_price_train = data.get_all_close()
# tickers_close_returns = ((tickers_close_price_train/tickers_close_price_train.shift(1)).dropna() - 1)
# exp_returns = mu(tickers_close_returns, tickers, 6 * 20)
# print(exp_returns.get_mu())
# print('_________________\n')
