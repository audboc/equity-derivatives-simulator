from math import exp, sqrt

import numpy as np


def simulate_gbm_path(S0, mu, sigma, T, n_steps, q=0, seed=None):
    """
    Simulates a path of the underlying via geometric Brownian motion:

        S(t+dt) = S(t) * exp[(mu - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z]

    S0 : starting spot
    mu : expected return (r under the risk-neutral measure, or a real-world return for P&L)
    sigma : realized volatility of the simulation
    T : total horizon in years
    n_steps : number of time steps
    q : continuous dividend yield

    Returns a numpy array of size n_steps+1 (S0 included as the first entry).
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

    # A simple path, to see what the result looks like
    path = simulate_gbm_path(S0=100, mu=0.05, sigma=0.20, T=1, n_steps=252, seed=42)

    print(f"Starting spot: {path[0]:.2f}")
    print(f"Final spot   : {path[-1]:.2f}")
    print(f"Number of points: {len(path)}")

    # Statistical check: over many paths, the average final spot should
    # converge to S0 * exp((mu - q) * T) (a property of GBM)
    n_paths = 20_000
    finals = [simulate_gbm_path(S0=100, mu=0.05, sigma=0.20, T=1, n_steps=252)[-1] for _ in range(n_paths)]

    empirical_mean = np.mean(finals)
    theoretical_mean = 100 * exp(0.05 * 1)

    print(f"\nOver {n_paths} paths:")
    print(f"Empirical mean of final spots : {empirical_mean:.2f}")
    print(f"Theoretical mean (S0 * exp(mu*T)): {theoretical_mean:.2f}")
