from math import log, sqrt, exp
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma, q=0):
    """
    Calcule le prix théorique d'un call européen
    avec Black-Scholes et dividende continu.

    S : prix actuel du sous-jacent
    K : strike
    T : maturité en années
    r : taux d'intérêt annuel
    sigma : volatilité annuelle
    q : rendement de dividende annuel
    """

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

    d2 = d1 - sigma * sqrt(T)

    call_price = (S * exp(-q * T) * norm.cdf(d1)- K * exp(-r * T) * norm.cdf(d2))

    return call_price


def black_scholes_put(S, K, T, r, sigma, q=0):
    """
    Calcule le prix théorique d'un put européen
    avec Black-Scholes et dividende continu.

    S : prix actuel du sous-jacent
    K : strike
    T : maturité en années
    r : taux d'intérêt annuel
    sigma : volatilité annuelle
    q : rendement de dividende annuel
    """

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

    d2 = d1 - sigma * sqrt(T)

    put_price = (K * exp(-r * T) * norm.cdf(-d2)- S * exp(-q * T) * norm.cdf(-d1)) #c'est -d1 et -d2 qui changent p

    return put_price


if __name__ == "__main__":

    # Exemple sans dividende

    price_without_dividend = black_scholes_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20
    )

    print(f"Prix du call sans dividende : {price_without_dividend:.2f} €")


    # Exemple avec un rendement de dividende de 2 %

    price_with_dividend = black_scholes_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        q=0.02
    )

    print(f"Prix du call avec dividende : {price_with_dividend:.2f} €")


    # Exemple de put sans dividende

    put_price_without_dividend = black_scholes_put(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20
    )

    print(f"Prix du put sans dividende : "f"{put_price_without_dividend:.2f} €")


    # Exemple de put avec un dividende de 2 %

    put_price_with_dividend = black_scholes_put(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        q=0.02
    )

    print( f"Prix du put avec dividende : "f"{put_price_with_dividend:.2f} €")


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
