from math import exp, sqrt

import numpy as np


def simulate_gbm_path(S0, mu, sigma, T, n_steps, q=0, seed=None):
    """
    Simule une trajectoire du sous-jacent par mouvement brownien géométrique :

        S(t+dt) = S(t) * exp[(mu - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z]

    S0 : spot de départ
    mu : rendement attendu (r en univers risque-neutre, ou un rendement réel pour du P&L)
    sigma : volatilité réalisée de la simulation
    T : horizon total en années
    n_steps : nombre de pas de temps
    q : rendement de dividende continu

    Retourne un tableau numpy de taille n_steps+1 (S0 inclus en première position).
    """

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    prices = np.empty(n_steps + 1)
    prices[0] = S0

    for t in range(1, n_steps + 1):
        Z = rng.standard_normal()
        prices[t] = prices[t - 1] * exp((mu - q - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * Z)

    return prices


if __name__ == "__main__":

    # Une trajectoire simple, pour voir à quoi ressemble le résultat
    path = simulate_gbm_path(S0=100, mu=0.05, sigma=0.20, T=1, n_steps=252, seed=42)

    print(f"Spot de départ : {path[0]:.2f}")
    print(f"Spot final     : {path[-1]:.2f}")
    print(f"Nombre de points : {len(path)}")

    # Vérification statistique : sur beaucoup de trajectoires, la moyenne des spots
    # finaux doit converger vers S0 * exp((mu - q) * T) (propriété du GBM)
    n_paths = 20_000
    finals = [simulate_gbm_path(S0=100, mu=0.05, sigma=0.20, T=1, n_steps=252)[-1] for _ in range(n_paths)]

    empirical_mean = np.mean(finals)
    theoretical_mean = 100 * exp(0.05 * 1)

    print(f"\nSur {n_paths} trajectoires :")
    print(f"Moyenne empirique des spots finaux : {empirical_mean:.2f}")
    print(f"Moyenne théorique (S0 * exp(mu*T))  : {theoretical_mean:.2f}")
