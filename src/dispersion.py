from math import sqrt

import numpy as np
import pandas as pd

from market_data import fetch_price_history, compute_log_returns, compute_realized_volatility


def fetch_basket_prices(tickers, period="2y", interval="1d"):
    """
    Downloads the price history of several stocks and aligns them into a single
    DataFrame (one column per ticker), keeping only the dates common to all
    tickers.
    """

    prices = {ticker: fetch_price_history(ticker, period=period, interval=interval) for ticker in tickers}
    return pd.DataFrame(prices).dropna()


def compute_correlation_matrix(log_returns_df):
    """
    Computes the correlation matrix of returns between every pair of stocks
    in the basket.
    """

    return log_returns_df.corr()


def compute_basket_level(prices_df, weights):
    """
    Builds the level of a "mini-index" = weighted basket of stocks. Each stock
    is first indexed to 100 at the start date (so that its dollar price doesn't
    distort the weight it's given), then the indexed levels are summed using
    the `weights`.

    weights : dict {ticker: weight}, must sum to 1.
    """

    indexed = 100 * prices_df / prices_df.iloc[0]
    weight_series = pd.Series(weights)[prices_df.columns]

    return (indexed * weight_series).sum(axis=1)


def compute_basket_volatility_formula(vols, corr_matrix, weights):
    """
    Reconstructs the basket's annualized volatility "from the bottom up", from
    the individual vols + the correlation matrix + the weights, via the
    variance formula for a weighted sum: σ_basket² = wᵀ Σ w, with Σ_ij = σ_i σ_j ρ_ij
    (covariance matrix built from the vols and the correlation).

    vols : dict {ticker: annualized vol}
    corr_matrix : correlation DataFrame (same tickers as vols/weights)
    weights : dict {ticker: weight}
    """

    tickers = corr_matrix.columns
    w = np.array([weights[t] for t in tickers])
    sigma = np.array([vols[t] for t in tickers])

    cov_matrix = np.outer(sigma, sigma) * corr_matrix.loc[tickers, tickers].to_numpy()
    basket_variance = w @ cov_matrix @ w

    return sqrt(basket_variance)


def analyze_dispersion(tickers, weights=None, period="2y", trading_days=252):
    """
    Full realized-dispersion pipeline on a stock basket:
      1. downloads stock prices and computes each one's realized vol + their
         correlation matrix
      2. builds the "mini-index" level (weighted basket) and computes its
         realized vol directly on its own series
      3. reconstructs that same vol "via the formula" (individual vols + correlation),
         to check that the two methods agree (round-trip)

    weights : dict {ticker: weight}, defaults to equal-weighted if not provided.

    Returns a dict with all the intermediate quantities (individual vols,
    correlation matrix, basket series, direct vol vs formula vol, weighted
    average of individual vols), ready to be displayed or plotted.
    """

    if weights is None:
        weights = {ticker: 1 / len(tickers) for ticker in tickers}

    prices_df = fetch_basket_prices(tickers, period=period)
    log_returns_df = compute_log_returns(prices_df)

    vols = {ticker: compute_realized_volatility(log_returns_df[ticker], trading_days=trading_days)
            for ticker in tickers}
    corr_matrix = compute_correlation_matrix(log_returns_df)

    basket_prices = compute_basket_level(prices_df, weights)
    basket_log_returns = compute_log_returns(basket_prices)
    basket_vol_direct = compute_realized_volatility(basket_log_returns, trading_days=trading_days)

    basket_vol_formula = compute_basket_volatility_formula(vols, corr_matrix, weights)

    weighted_avg_component_vol = sum(weights[t] * vols[t] for t in tickers)

    return {
        "tickers": tickers,
        "weights": weights,
        "vols": vols,
        "corr_matrix": corr_matrix,
        "basket_prices": basket_prices,
        "basket_vol_direct": basket_vol_direct,
        "basket_vol_formula": basket_vol_formula,
        "weighted_avg_component_vol": weighted_avg_component_vol,
    }


if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    result = analyze_dispersion(tickers)

    print("Annualized realized vols (2 years):")
    for ticker in tickers:
        print(f"  {ticker}: {result['vols'][ticker]:.2%}")

    print("\nCorrelation matrix:")
    print(result["corr_matrix"].round(2))

    print(f"\nBasket vol (direct, on the real basket series): {result['basket_vol_direct']:.2%}")
    print(f"Basket vol (formula, vols + correlation)       : {result['basket_vol_formula']:.2%}")
    print(f"Weighted average of individual vols             : {result['weighted_avg_component_vol']:.2%}")
