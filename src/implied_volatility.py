from scipy.optimize import brentq

from black_scholes import black_scholes_call, black_scholes_put


def implied_volatility_call(market_price, S, K, T, r, q=0):
    """
    Recovers the implied volatility of a European call from its market price,
    by searching for the sigma that makes Black-Scholes match that price (Brent's method).
    """

    def price_difference(sigma):
        return black_scholes_call(S, K, T, r, sigma, q) - market_price

    return brentq(price_difference, 1e-6, 5)


def implied_volatility_put(market_price, S, K, T, r, q=0):
    """
    Recovers the implied volatility of a European put from its market price,
    by searching for the sigma that makes Black-Scholes match that price (Brent's method).
    """

    def price_difference(sigma):
        return black_scholes_put(S, K, T, r, sigma, q) - market_price

    return brentq(price_difference, 1e-6, 5)


if __name__ == "__main__":

    # We start from a known vol, compute the price, then recover the vol from the price.
    # If everything goes well, the recovered vol should be nearly identical to the starting vol.

    true_sigma = 0.25

    market_call_price = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=true_sigma)
    recovered_call_vol = implied_volatility_call(market_call_price, S=100, K=100, T=1, r=0.05)

    print(f"Starting vol (call) : {true_sigma:.4f}")
    print(f"Recovered vol (call): {recovered_call_vol:.4f}")

    market_put_price = black_scholes_put(S=100, K=100, T=1, r=0.05, sigma=true_sigma)
    recovered_put_vol = implied_volatility_put(market_put_price, S=100, K=100, T=1, r=0.05)

    print(f"Starting vol (put)  : {true_sigma:.4f}")
    print(f"Recovered vol (put) : {recovered_put_vol:.4f}")
