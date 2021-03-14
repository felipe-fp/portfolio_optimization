import numpy as np
from arch import arch_model

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
        for p_ in range(1, 20):
            for q_ in range(1, 20):
                model = arch_model(time_series, p = p_, q = q_, mean = 'constant')
                model_fit = model.fit()
                current_perf = model_fit.aic
                if current_perf < best_perf:
                    var = (model_fit.forecast(horizon = time_horizon).variance.values)[-1][0]
                    best_perf = current_perf
        return var
    def compute_sigma(self, sigma, P):
        return sigma @ P @ sigma
    def get_variance(self):
        return self.variance
    def get_correlation(self):
        return self.corr
    def get_sigma(self):
        return self.sigma