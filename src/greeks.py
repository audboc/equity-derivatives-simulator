from math import log, sqrt, exp
from scipy.stats import norm


def calculate_d1(S, K, T, r, sigma, q=0):
    """
    Calcule d1 dans le modèle de Black-Scholes.
    """

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

    return d1


def call_delta(S, K, T, r, sigma, q=0):
    """
    Calcule le delta d'un call européen (sensibilité du produit au spot)
    """

    d1 = calculate_d1(S, K, T, r, sigma, q)

    delta = exp(-q * T) * norm.cdf(d1)

    return delta


def put_delta(S, K, T, r, sigma, q=0):
    """
    Calcule le delta d'un put européen (sensibilité du produit au spot)
    """

    d1 = calculate_d1(S, K, T, r, sigma, q)

    delta = exp(-q * T) * (norm.cdf(d1) - 1)

    return delta


def gamma(S, K, T, r, sigma, q=0):
    """
    Calcule le gamma d'une option européenne (sensibilité du delta au spot).
    Identique pour un call et un put.
    """

    d1 = calculate_d1(S, K, T, r, sigma, q)

    return exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))


def vega(S, K, T, r, sigma, q=0):
    """
    Calcule le vega d'une option européenne (sensibilité du prix à la volatilité).
    Identique pour un call et un put.
    """

    d1 = calculate_d1(S, K, T, r, sigma, q)

    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T)


def call_theta(S, K, T, r, sigma, q=0):
    """
    Calcule le theta d'un call européen (sensibilité du prix au temps qui passe).
    Theta annuel (non divisé par 365).
    """

    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * sqrt(T)

    term1 = -(S * exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt(T))
    term2 = -r * K * exp(-r * T) * norm.cdf(d2)
    term3 = q * S * exp(-q * T) * norm.cdf(d1)

    return term1 + term2 + term3


def put_theta(S, K, T, r, sigma, q=0):
    """
    Calcule le theta d'un put européen (sensibilité du prix au temps qui passe).
    Theta annuel (non divisé par 365).
    """

    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * sqrt(T)

    term1 = -(S * exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt(T))
    term2 = r * K * exp(-r * T) * norm.cdf(-d2)
    term3 = -q * S * exp(-q * T) * norm.cdf(-d1)

    return term1 + term2 + term3


def call_rho(S, K, T, r, sigma, q=0):
    """
    Calcule le rho d'un call européen (sensibilité du prix au taux d'intérêt).
    """

    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * sqrt(T)

    return K * T * exp(-r * T) * norm.cdf(d2)


def put_rho(S, K, T, r, sigma, q=0):
    """
    Calcule le rho d'un put européen (sensibilité du prix au taux d'intérêt).
    """

    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * sqrt(T)

    return -K * T * exp(-r * T) * norm.cdf(-d2)


if __name__ == "__main__":

    call_delta_value = call_delta(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)
    put_delta_value = put_delta(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)

    print(f"Delta du call : {call_delta_value:.4f}")
    print(f"Delta du put : {put_delta_value:.4f}")

    gamma_value = gamma(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)

    print(f"Gamma : {gamma_value:.4f}")

    vega_value = vega(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)

    print(f"Vega : {vega_value:.4f}")

    call_theta_value = call_theta(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)
    put_theta_value = put_theta(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)

    print(f"Theta du call : {call_theta_value:.4f}")
    print(f"Theta du put : {put_theta_value:.4f}")

    call_rho_value = call_rho(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)
    put_rho_value = put_rho(S=100, K=100, T=1, r=0.05, sigma=0.20, q=0)

    print(f"Rho du call : {call_rho_value:.4f}")
    print(f"Rho du put : {put_rho_value:.4f}")
