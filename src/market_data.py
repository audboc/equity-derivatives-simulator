from math import sqrt

import numpy as np
import yfinance as yf


def fetch_price_history(ticker, period="2y", interval="1d"):
    """
    Downloads an asset's price history via yfinance (Yahoo Finance) and returns
    the closing prices (adjusted for dividends/splits) as a pandas Series,
    indexed by date.

    ticker : yfinance symbol (e.g. "^GSPC" for the S&P 500, "AAPL" for Apple)
    period : length of history requested (e.g. "1y", "2y", "5y", "max")
    interval : data frequency (e.g. "1d" for daily)
    """

    data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data retrieved for {ticker} (invalid ticker or no network connection).")

    return data["Close"]


def compute_log_returns(prices):
    """
    Computes the logarithmic returns of a price series: r_t = ln(S_t / S_{t-1}).
    """

    return np.log(prices / prices.shift(1)).dropna()


def compute_realized_volatility(log_returns, trading_days=252):
    """
    Computes the annualized realized volatility over the whole period: standard
    deviation of the log returns, annualized by the square root of trading days per year.
    """

    return log_returns.std() * sqrt(trading_days)


def compute_rolling_realized_volatility(log_returns, window=21, trading_days=252):
    """
    Computes the annualized realized volatility over a rolling window (21 days by
    default, roughly one month of trading), to visualize how it evolves over time.
    """

    return log_returns.rolling(window).std() * sqrt(trading_days)


if __name__ == "__main__":

    # S&P 500 and Apple: the same functions work with any yfinance ticker
    # (e.g. "MSFT", "NVDA", "AMZN", "META", "JPM", ...), just change the ticker.

    for ticker in ["^GSPC", "AAPL"]:
        prices = fetch_price_history(ticker, period="2y")
        log_returns = compute_log_returns(prices)
        realized_vol = compute_realized_volatility(log_returns)

        print(f"{ticker}: annualized realized vol over 2 years = {realized_vol:.2%}")
