import numpy as np
from arch import arch_model
import pandas as pd

class Sigma():
    def __init__(self, ticker_series, time_horizon):
        self.variance = np.diag([self.compute_variance(ticker_series[ticker_time_series],time_horizon) for ticker_time_series in ticker_series])
        self.corr = self.compute_correlation(ticker_series)
        self.sigma = self.compute_sigma(self.variance, self.corr)
    def compute_correlation(self, ticker_series):
        P = ticker_series.corr(method = 'pearson').to_numpy()
        return P
    def compute_variance(self, time_series, time_horizon):
        best_perf = np.inf
        for p_ in range(1, 10):
            for q_ in range(1, 10):
                model = arch_model(time_series, p = p_, q = q_, mean = 'constant')
                model_fit = model.fit()
                current_perf = model_fit.aic
                if current_perf < best_perf:
                    var = (model_fit.forecast(horizon = time_horizon).variance.values)[-1][0]
                    best_perf = current_perf
        return var
    def compute_sigma(self, sigma, P):
        return 252*np.sqrt(sigma) @ P @ np.sqrt(sigma)
    def get_variance(self):
        return self.variance
    def get_correlation(self):
        return self.corr
    def get_sigma(self):
        return self.sigma

# a = np.random.normal(0,3, 10000)
# b = np.random.normal(-1,2.2, 10000)
# c  = np.random.normal(2,0.3, 10000)

# df = pd.DataFrame({'A':a, 'BB':b, 'C':c, 'D': a})
# obj = Sigma(df, 1)
# print('Variance: \n')
# print(obj.get_variance())
# print('\n -------- \n')
# print('Correlation: \n')
# print(obj.get_correlation())
# print('\n -------- \n')
# print('Covariance: \n')
# print(obj.get_sigma())
# print('\n')
# print(df.cov())
# print('\n -------- \n')


