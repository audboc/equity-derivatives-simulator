import pytest

from greeks import call_delta, put_delta, gamma, vega, call_rho, put_rho

from params import PARAM_COMBOS


@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAM_COMBOS)
def test_call_delta_is_between_0_and_1(S, K, T, r, sigma, q):
    assert 0 <= call_delta(S, K, T, r, sigma, q) <= 1


@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAM_COMBOS)
def test_put_delta_is_between_minus1_and_0(S, K, T, r, sigma, q):
    assert -1 <= put_delta(S, K, T, r, sigma, q) <= 0


@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAM_COMBOS)
def test_gamma_is_always_positive(S, K, T, r, sigma, q):
    assert gamma(S, K, T, r, sigma, q) > 0


@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAM_COMBOS)
def test_vega_is_always_positive(S, K, T, r, sigma, q):
    assert vega(S, K, T, r, sigma, q) > 0


@pytest.mark.parametrize("S, K, T, r, sigma, q", PARAM_COMBOS)
def test_call_rho_positive_and_put_rho_negative(S, K, T, r, sigma, q):
    assert call_rho(S, K, T, r, sigma, q) > 0
    assert put_rho(S, K, T, r, sigma, q) < 0
