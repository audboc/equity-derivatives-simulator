from math import log, sqrt, exp
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma, q=0):
    """
    Computes the theoretical price of a European call
    with Black-Scholes and a continuous dividend.

    S : current price of the underlying
    K : strike
    T : maturity in years
    r : annual interest rate
    sigma : annual volatility
    q : annual dividend yield
    """

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

    d2 = d1 - sigma * sqrt(T)

    call_price = (S * exp(-q * T) * norm.cdf(d1)- K * exp(-r * T) * norm.cdf(d2))

    return call_price


def black_scholes_put(S, K, T, r, sigma, q=0):
    """
    Computes the theoretical price of a European put
    with Black-Scholes and a continuous dividend.

    S : current price of the underlying
    K : strike
    T : maturity in years
    r : annual interest rate
    sigma : annual volatility
    q : annual dividend yield
    """

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

    d2 = d1 - sigma * sqrt(T)

    put_price = (K * exp(-r * T) * norm.cdf(-d2)- S * exp(-q * T) * norm.cdf(-d1)) # -d1 and -d2 are what changes

    return put_price


if __name__ == "__main__":

    # Example without dividend

    price_without_dividend = black_scholes_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20
    )

    print(f"Call price without dividend: {price_without_dividend:.2f} $")


    # Example with a 2% dividend yield

    price_with_dividend = black_scholes_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        q=0.02
    )

    print(f"Call price with dividend: {price_with_dividend:.2f} $")


    # Put example without dividend

    put_price_without_dividend = black_scholes_put(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20
    )

    print(f"Put price without dividend: "f"{put_price_without_dividend:.2f} $")


    # Put example with a 2% dividend

    put_price_with_dividend = black_scholes_put(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        q=0.02
    )

    print( f"Put price with dividend: "f"{put_price_with_dividend:.2f} $")


    # Put-Call Parity (verification of black & scholes results) => C - P = S*exp(-q*T) - K*exp(-r*T)#

    call_price = black_scholes_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        q=0
    )

    put_price = black_scholes_put(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        q=0
    )

    left_side = call_price - put_price
    right_side = 100 * exp(-0 * 1) - 100 * exp(-0.05 * 1)

    print(f"C - P : {left_side:.4f}")
    print(f"S - K exp(-rT) : {right_side:.4f}")
