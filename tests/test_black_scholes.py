"""Tests for volsurf.black_scholes: benchmark, parity, Greeks and identities."""

import numpy as np
import pytest

from volsurf.black_scholes import (
    black76_price,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta,
    bs_vega,
)

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

S0 = 100.0
RATES = [(0.0, 0.0), (0.04, 0.013)]
FLAGS = [True, False]


def _greek_grid(r, q):
    """Moderate grid (Greeks well away from zero) with strikes set off the forward."""
    k_over_f, T, sigma = np.meshgrid(
        [0.9, 0.95, 1.0, 1.05, 1.1],
        [21 / 252, 0.25, 1.0, 2.0],
        [0.1, 0.18, 0.4],
        indexing="ij",
    )
    K = k_over_f * S0 * np.exp((r - q) * T)
    return K, T, sigma


def _wide_grid():
    """Wide grid for parity: deep ITM/OTM, 7 days to 5 years, high vol, nonzero r and q."""
    return np.meshgrid(
        [50.0, 80.0, 95.0, 100.0, 105.0, 120.0, 200.0],
        [7 / 365, 21 / 252, 0.5, 2.0, 5.0],
        [0.05, 0.18, 0.4, 0.8],
        [0.0, 0.04, 0.1],
        [0.0, 0.013, 0.03],
        indexing="ij",
    )


def _fd1(f, h):
    """Five-point central first derivative of f(u) at u = 0; error O(h^4)."""
    return (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)


# ---------------------------------------------------------------------------
# Benchmark (DESIGN.md section 2)
# ---------------------------------------------------------------------------

def test_atm_benchmark():
    # DESIGN.md values to 6 decimals: C0 = 2.072732, vega = 11.512585,
    # vega * sigma = 2.072265. Asserted here at full precision.
    T = 21 / 252
    c0 = bs_price(S0, S0, T, 0.0, 0.0, 0.18, is_call=True)
    vega = bs_vega(S0, S0, T, 0.0, 0.0, 0.18)
    print(f"C0 = {c0:.12f}, vega = {vega:.12f}, vega*sigma = {vega * 0.18:.12f}")
    assert c0 == pytest.approx(2.072731712, abs=1e-9)
    assert vega == pytest.approx(11.512585496, abs=1e-9)


# ---------------------------------------------------------------------------
# Put-call parity
# ---------------------------------------------------------------------------

def test_put_call_parity_forward():
    K, T, sigma, r, q = _wide_grid()
    F = S0 * np.exp((r - q) * T)
    D = np.exp(-r * T)
    call = black76_price(F, K, T, D, sigma, is_call=True)
    put = black76_price(F, K, T, D, sigma, is_call=False)
    np.testing.assert_allclose(call - put, D * (F - K), rtol=0, atol=1e-12)


def test_put_call_parity_spot():
    K, T, sigma, r, q = _wide_grid()
    call = bs_price(S0, K, T, r, q, sigma, is_call=True)
    put = bs_price(S0, K, T, r, q, sigma, is_call=False)
    expected = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(call - put, expected, rtol=0, atol=1e-12)


def test_spot_wrapper_matches_black76():
    K, T, sigma, r, q = _wide_grid()
    F = S0 * np.exp((r - q) * T)
    D = np.exp(-r * T)
    for is_call in FLAGS:
        np.testing.assert_array_equal(
            bs_price(S0, K, T, r, q, sigma, is_call),
            black76_price(F, K, T, D, sigma, is_call),
        )


# ---------------------------------------------------------------------------
# Greeks against finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r, q", RATES)
@pytest.mark.parametrize("is_call", FLAGS)
def test_delta_finite_difference(r, q, is_call):
    K, T, sigma = _greek_grid(r, q)
    fd = _fd1(lambda u: bs_price(S0 + u, K, T, r, q, sigma, is_call), 1e-2)
    np.testing.assert_allclose(bs_delta(S0, K, T, r, q, sigma, is_call), fd, rtol=1e-6, atol=0)


@pytest.mark.parametrize("r, q", RATES)
@pytest.mark.parametrize("is_call", FLAGS)
def test_gamma_finite_difference(r, q, is_call):
    # Difference of the analytic delta (itself checked against prices above):
    # a second difference of prices is limited by rounding at about 2e-6.
    K, T, sigma = _greek_grid(r, q)
    fd = _fd1(lambda u: bs_delta(S0 + u, K, T, r, q, sigma, is_call), 1e-2)
    np.testing.assert_allclose(bs_gamma(S0, K, T, r, q, sigma), fd, rtol=1e-6, atol=0)


@pytest.mark.parametrize("r, q", RATES)
@pytest.mark.parametrize("is_call", FLAGS)
def test_vega_finite_difference(r, q, is_call):
    K, T, sigma = _greek_grid(r, q)
    fd = _fd1(lambda u: bs_price(S0, K, T, r, q, sigma + u, is_call), 1e-4)
    np.testing.assert_allclose(bs_vega(S0, K, T, r, q, sigma), fd, rtol=1e-6, atol=0)


@pytest.mark.parametrize("r, q", RATES)
@pytest.mark.parametrize("is_call", FLAGS)
def test_theta_finite_difference(r, q, is_call):
    K, T, sigma = _greek_grid(r, q)
    fd = -_fd1(lambda u: bs_price(S0, K, T + u, r, q, sigma, is_call), 1e-4)
    np.testing.assert_allclose(bs_theta(S0, K, T, r, q, sigma, is_call), fd, rtol=1e-6, atol=0)


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

def test_gamma_vega_identity():
    # Deep ITM/OTM corners underflow vega to 0 or subnormals, where a relative
    # check says nothing; keep points with vega > 1e-12 and require most remain.
    K, T, sigma, r, q = _wide_grid()
    vega = bs_vega(S0, K, T, r, q, sigma)
    lhs = bs_gamma(S0, K, T, r, q, sigma) * S0**2 * sigma * T
    keep = vega > 1e-12
    assert keep.sum() >= 1000
    np.testing.assert_allclose(lhs[keep], vega[keep], rtol=1e-10, atol=0)


@pytest.mark.parametrize("r, q", RATES)
@pytest.mark.parametrize("is_call", FLAGS)
def test_theta_satisfies_pde(r, q, is_call):
    # theta + (r - q)·S·delta + ½·sigma²·S²·gamma - r·V = 0
    K, T, sigma = _greek_grid(r, q)
    residual = (
        bs_theta(S0, K, T, r, q, sigma, is_call)
        + (r - q) * S0 * bs_delta(S0, K, T, r, q, sigma, is_call)
        + 0.5 * sigma**2 * S0**2 * bs_gamma(S0, K, T, r, q, sigma)
        - r * bs_price(S0, K, T, r, q, sigma, is_call)
    )
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------

def test_vectorised_mixed_flags():
    S = np.array([[95.0], [100.0], [105.0]])  # shape (3, 1)
    K = np.array([90.0, 100.0, 110.0, 120.0])  # shape (4,)
    is_call = np.array([True, False, True, False])
    T, r, q, sigma = 0.5, 0.04, 0.013, 0.2
    funcs = [
        lambda *a: bs_price(*a[:6], a[6]),
        lambda *a: bs_delta(*a[:6], a[6]),
        lambda *a: bs_theta(*a[:6], a[6]),
        lambda *a: bs_gamma(*a[:6]),
        lambda *a: bs_vega(*a[:6]),
    ]
    for f in funcs:
        out = f(S, K, T, r, q, sigma, is_call)
        assert out.shape == (3, 4)
        for i in range(3):
            for j in range(4):
                scalar = f(S[i, 0], K[j], T, r, q, sigma, bool(is_call[j]))
                assert out[i, j] == pytest.approx(scalar, rel=1e-15, abs=0)
