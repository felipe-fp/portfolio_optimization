from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

class mu():
    def __init__(self, ticker_series):
        self.expected_returns = [self.fit_ARIMA(ticker_series[ticker_time_series]) for ticker_time_series in ticker_series] 
    def log_stationarity(self, time_series):
        log_series = np.log(time_series)
        base_point = log_series.values[-1]
        new_series = (log_series - log_series.shift(1)).dropna()
        return new_series, base_point
    def fit_ARIMA(self, time_series):
        new_series, base_point = self.log_stationarity(time_series)
        best_perf = np.inf
        for p_ in range(1,5):
            for d_ in range(0,3):
                for q_ in range(1,5):
                    model = ARIMA(endog=new_series, order = (p_,d_,q_), enforce_stationarity=True)
                    try:
                        model_fit = model.fit()
                        current_perf = model_fit.aic
                        if current_perf < best_perf:
                            mu = model_fit.forecast().values[0]
                            best_perf = current_perf
                    except:
                        pass
        return np.exp(mu + base_point)
    def get_mu(self):
        return self.expected_returns