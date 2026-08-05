from math import sqrt

import numpy as np
import yfinance as yf


def fetch_price_history(ticker, period="2y", interval="1d"):
    """
    Télécharge l'historique de prix d'un actif via yfinance (Yahoo Finance) et retourne
    les prix de clôture (ajustés des dividendes/splits) sous forme de Series pandas,
    indexée par date.

    ticker : symbole yfinance (ex : "^GSPC" pour le S&P 500, "AAPL" pour Apple)
    period : durée d'historique demandée (ex : "1y", "2y", "5y", "max")
    interval : fréquence des données (ex : "1d" pour quotidien)
    """

    data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

    if data.empty:
        raise ValueError(f"Aucune donnée récupérée pour {ticker} (ticker invalide ou pas de connexion réseau).")

    return data["Close"]


def compute_log_returns(prices):
    """
    Calcule les rendements logarithmiques d'une série de prix : r_t = ln(S_t / S_{t-1}).
    """

    return np.log(prices / prices.shift(1)).dropna()


def compute_realized_volatility(log_returns, trading_days=252):
    """
    Calcule la volatilité réalisée annualisée sur toute la période : écart-type des
    rendements logarithmiques, annualisé par racine(nombre de jours de trading par an).
    """

    return log_returns.std() * sqrt(trading_days)


def compute_rolling_realized_volatility(log_returns, window=21, trading_days=252):
    """
    Calcule la volatilité réalisée annualisée en fenêtre glissante (par défaut 21 jours,
    environ un mois de trading), pour visualiser son évolution dans le temps.
    """

    return log_returns.rolling(window).std() * sqrt(trading_days)


if __name__ == "__main__":

    # S&P 500 et Apple : les mêmes fonctions marchent avec n'importe quel ticker yfinance
    # (ex : "MSFT", "NVDA", "AMZN", "META", "JPM", ...), il suffit de changer le ticker.

    for ticker in ["^GSPC", "AAPL"]:
        prices = fetch_price_history(ticker, period="2y")
        log_returns = compute_log_returns(prices)
        realized_vol = compute_realized_volatility(log_returns)

        print(f"{ticker} : vol réalisée annualisée sur 2 ans = {realized_vol:.2%}")
