from scipy.optimize import brentq

from black_scholes import black_scholes_call, black_scholes_put


def implied_volatility_call(market_price, S, K, T, r, q=0):
    """
    Retrouve la volatilité implicite d'un call européen à partir de son prix de marché,
    en cherchant le sigma qui fait coïncider Black-Scholes avec ce prix (méthode de Brent).
    """

    def price_difference(sigma):
        return black_scholes_call(S, K, T, r, sigma, q) - market_price

    return brentq(price_difference, 1e-6, 5)


def implied_volatility_put(market_price, S, K, T, r, q=0):
    """
    Retrouve la volatilité implicite d'un put européen à partir de son prix de marché,
    en cherchant le sigma qui fait coïncider Black-Scholes avec ce prix (méthode de Brent).
    """

    def price_difference(sigma):
        return black_scholes_put(S, K, T, r, sigma, q) - market_price

    return brentq(price_difference, 1e-6, 5)


if __name__ == "__main__":

    # On part d'une vol connue, on calcule le prix, puis on retrouve la vol à partir du prix.
    # Si tout va bien, la vol retrouvée doit être quasiment identique à la vol de départ.

    true_sigma = 0.25

    market_call_price = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=true_sigma)
    recovered_call_vol = implied_volatility_call(market_call_price, S=100, K=100, T=1, r=0.05)

    print(f"Vol de départ (call) : {true_sigma:.4f}")
    print(f"Vol retrouvée (call) : {recovered_call_vol:.4f}")

    market_put_price = black_scholes_put(S=100, K=100, T=1, r=0.05, sigma=true_sigma)
    recovered_put_vol = implied_volatility_put(market_put_price, S=100, K=100, T=1, r=0.05)

    print(f"Vol de départ (put)  : {true_sigma:.4f}")
    print(f"Vol retrouvée (put)  : {recovered_put_vol:.4f}")
