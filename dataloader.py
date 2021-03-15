import yfinance
import pandas as pd

class Dataloader():
    def __init__(self, period, tickers_list):

        self.period = period
        self.tickers = tickers_list

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
        return self.close_prices
    def get_open(self):
        return self.open_prices
    def get_volume(self):
        return self.volume

# tickers = ['AAPL', 'GOOG', 'TSLA']
# period = '12mo'
# data = Dataloader('12mo', tickers)
# tickers_close_price = data.get_close()
# print(tickers_close_price)