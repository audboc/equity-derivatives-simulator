from math import sqrt

import numpy as np
import pandas as pd

from market_data import fetch_price_history, compute_log_returns, compute_realized_volatility


def fetch_basket_prices(tickers, period="2y", interval="1d"):
    """
    Télécharge l'historique de prix de plusieurs actions et les aligne dans un seul
    DataFrame (une colonne par ticker), en ne gardant que les dates communes à tous
    les tickers.
    """

    prices = {ticker: fetch_price_history(ticker, period=period, interval=interval) for ticker in tickers}
    return pd.DataFrame(prices).dropna()


def compute_correlation_matrix(log_returns_df):
    """
    Calcule la matrice de corrélation des rendements entre toutes les paires d'actions
    du panier.
    """

    return log_returns_df.corr()


def compute_basket_level(prices_df, weights):
    """
    Construit le niveau d'un "mini-indice" = panier pondéré des actions. Chaque action
    est d'abord indexée à 100 à la date de départ (pour que son prix en dollars ne
    fausse pas le poids qu'on lui donne), puis les niveaux indexés sont sommés avec
    les poids `weights`.

    weights : dict {ticker: poids}, doit sommer à 1.
    """

    indexed = 100 * prices_df / prices_df.iloc[0]
    weight_series = pd.Series(weights)[prices_df.columns]

    return (indexed * weight_series).sum(axis=1)


def compute_basket_volatility_formula(vols, corr_matrix, weights):
    """
    Reconstruit la volatilité annualisée du panier "par le bas" (bottom-up), à partir
    des vols individuelles + de la matrice de corrélation + des poids, via la formule
    de variance d'une somme pondérée : σ_panier² = wᵀ Σ w, avec Σ_ij = σ_i σ_j ρ_ij
    (matrice de covariance construite depuis les vols et la corrélation).

    vols : dict {ticker: vol annualisée}
    corr_matrix : DataFrame de corrélation (mêmes tickers que vols/weights)
    weights : dict {ticker: poids}
    """

    tickers = corr_matrix.columns
    w = np.array([weights[t] for t in tickers])
    sigma = np.array([vols[t] for t in tickers])

    cov_matrix = np.outer(sigma, sigma) * corr_matrix.loc[tickers, tickers].to_numpy()
    basket_variance = w @ cov_matrix @ w

    return sqrt(basket_variance)


def analyze_dispersion(tickers, weights=None, period="2y", trading_days=252):
    """
    Pipeline complet de dispersion réalisée sur un panier d'actions :
      1. télécharge les prix des actions et calcule la vol réalisée de chacune + leur
         matrice de corrélation
      2. construit le niveau du "mini-indice" (panier pondéré) et calcule sa vol
         réalisée directement sur sa propre série
      3. reconstruit cette même vol "par la formule" (vols individuelles + corrélation),
         pour vérifier que les deux méthodes concordent (round-trip)

    weights : dict {ticker: poids}, par défaut équipondéré si non fourni.

    Retourne un dict avec toutes les quantités intermédiaires (vols individuelles,
    matrice de corrélation, série du panier, vol directe vs vol formule, moyenne
    pondérée des vols individuelles), prêtes à être affichées ou tracées.
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

    print("Vols réalisées annualisées (2 ans) :")
    for ticker in tickers:
        print(f"  {ticker} : {result['vols'][ticker]:.2%}")

    print("\nMatrice de corrélation :")
    print(result["corr_matrix"].round(2))

    print(f"\nVol du panier (directe, sur la vraie série du panier) : {result['basket_vol_direct']:.2%}")
    print(f"Vol du panier (formule, vols + corrélation)            : {result['basket_vol_formula']:.2%}")
    print(f"Moyenne pondérée des vols individuelles                : {result['weighted_avg_component_vol']:.2%}")
