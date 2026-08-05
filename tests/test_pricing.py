from math import exp

import pytest

from black_scholes import black_scholes_call, black_scholes_put

from params import PARAM_COMBOS


@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAM_COMBOS)
def test_call_and_put_prices_are_never_negative(S, K, T, r, sigma, q):
    assert black_scholes_call(S, K, T, r, sigma, q) >= 0
    assert black_scholes_put(S, K, T, r, sigma, q) >= 0


@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAM_COMBOS)
def test_put_call_parity(S, K, T, r, sigma, q):
    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)

    left_side = call_price - put_price
    right_side = S * exp(-q * T) - K * exp(-r * T)

    assert left_side == pytest.approx(right_side, abs=1e-8)


def test_call_price_increases_with_spot():
    K, T, r, sigma, q = 100, 1, 0.05, 0.20, 0
    spots = [70, 90, 100, 110, 130]

    prices = [black_scholes_call(S, K, T, r, sigma, q) for S in spots]

    assert prices == sorted(prices)


def test_put_price_decreases_with_spot():
    K, T, r, sigma, q = 100, 1, 0.05, 0.20, 0
    spots = [70, 90, 100, 110, 130]

    prices = [black_scholes_put(S, K, T, r, sigma, q) for S in spots]

    assert prices == sorted(prices, reverse=True)


@pytest.mark.parametrize("S, K, T, r, q", [(100, 100, 1, 0.05, 0), (90, 100, 0.5, 0.03, 0.02)])
def test_call_and_put_prices_increase_with_volatility(S, K, T, r, q):
    vols = [0.10, 0.20, 0.30, 0.50]

    call_prices = [black_scholes_call(S, K, T, r, sigma, q) for sigma in vols]
    put_prices = [black_scholes_put(S, K, T, r, sigma, q) for sigma in vols]

    assert call_prices == sorted(call_prices)
    assert put_prices == sorted(put_prices)
