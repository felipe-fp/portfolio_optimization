import yfinance
import pandas as pd
from datetime import timedelta

class Dataloader():
    def __init__(self, period, tickers_list, rebalacing_freq_months):

        self.period = period
        self.tickers = tickers_list
        self.rebalacing_freq_months = rebalacing_freq_months

        data_close = pd.DataFrame(yfinance.Ticker(tickers_list[0]).history(period = period).Close)
        data_close.columns = [tickers_list[0]]

        data_open = pd.DataFrame(yfinance.Ticker(tickers_list[0]).history(period = period).Open)
        data_open.columns = [tickers_list[0]]

        data_volume = pd.DataFrame(yfinance.Ticker(tickers_list[0]).history(period = period).Volume)
        data_volume.columns = [tickers_list[0]]

        for ticker in tickers_list[1:]:
            ticker_close = pd.DataFrame(yfinance.Ticker(ticker).history(period = period).Close)
            ticker_close.columns = [ticker]

            ticker_open = pd.DataFrame(yfinance.Ticker(ticker).history(period = period).Open)
            ticker_open.columns = [ticker]


            ticker_volume = pd.DataFrame(yfinance.Ticker(ticker).history(period = period).Volume)
            ticker_volume.columns = [ticker]

            data_close = data_close.join(ticker_close, how = 'outer')
            data_open = data_open.join(ticker_open, how = 'outer')
            data_volume = data_volume.join(ticker_volume, how = 'outer')

        self.close_prices = data_close
        self.open_prices = data_open
        self.volume = data_volume

    def get_close(self):
        start_index = self.close_prices.index[0]
        stop_index = self.close_prices.index[-1] - timedelta(days = self.rebalacing_freq_months)
        close_prices_for_rebalancing = []
        dates = []
        while start_index <= stop_index:
            end_rebalancing = start_index + timedelta(days = self.rebalacing_freq_months)
            close_df = self.close_prices.loc[start_index : end_rebalancing]
            close_prices_for_rebalancing.append(close_df)
            start_index = end_rebalancing
            dates.append(start_index)

        return dates, close_prices_for_rebalancing
        
    def get_all_close(self):
        return self.close_prices

    def get_open(self):
        start_index = self.open_prices.index[0]
        stop_index = self.open_prices.index[-1] - timedelta(days = self.rebalacing_freq_months)
        open_prices_for_rebalancing = []
        while start_index <= stop_index:
            end_rebalancing = start_index + timedelta(days = self.rebalacing_freq_months)
            open_df = self.open_prices.loc[start_index : end_rebalancing]
            open_prices_for_rebalancing.append(open_df)
            start_index = end_rebalancing

        return open_prices_for_rebalancing
    def get_volume(self):
        start_index = self.volume.index[0]
        stop_index = self.volume.index[-1] - timedelta(days = self.rebalacing_freq_months)
        volume_prices_for_rebalancing = []
        while start_index <= stop_index:
            end_rebalancing = start_index + timedelta(days = self.rebalacing_freq_months)
            volume_df = self.volume.loc[start_index : end_rebalancing]
            volume_prices_for_rebalancing.append(volume_df)
            start_index = end_rebalancing

        return volume_prices_for_rebalancing

# tickers = ['AAPL', 'GOOG', 'TSLA']
# period = '12mo'
# data = Dataloader('12mo', tickers,2)
# dates, tickers_close_price = data.get_close()
# print(tickers_close_price)
# print(dates)
# print('_______________\n')