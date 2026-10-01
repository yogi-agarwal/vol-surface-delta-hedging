"""Tests for volsurf.implied_vol: round trip, Newton demo, bounds and the forward vega."""

import numpy as np
import pytest

from volsurf.black_scholes import black76_price, black76_vega, bs_vega
from volsurf.implied_vol import black76_implied_vol, newton_manaster_koehler, newton_naive

# DESIGN.md section 6 Newton demo: a 15% out-of-the-money 14-day put.
DEMO = {"F": 100.0, "K": 85.0, "T": 14 / 365, "D": 1.0, "is_call": False}
DEMO_SIGMA = 0.40


def _demo_price():
    """The demo put priced at sigma = 0.40 exactly (0.05007364 to 8 decimals)."""
    return float(black76_price(DEMO["F"], DEMO["K"], DEMO["T"], DEMO["D"], DEMO_SIGMA, DEMO["is_call"]))


# ---------------------------------------------------------------------------
# Round trip (DESIGN.md section 6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [0.0, 0.04])
def test_round_trip_grid(rate):
    # Puts below F, calls at or above it. sigma is recovered within 1e-6 wherever the price is
    # above 0; the only exceptions are prices that underflow to exactly 0, which are skipped.
    T_days, k_over_f, sigma = np.meshgrid(
        [7, 14, 30, 91, 365],
        [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5],
        [0.05, 0.1, 0.2, 0.4, 0.8],
        indexing="ij",
    )
    T = T_days / 365
    F, D = 100.0, np.exp(-rate * T)
    K, is_call = k_over_f * F, k_over_f >= 1.0
    price = black76_price(F, K, T, D, sigma, is_call)
    implied = black76_implied_vol(price, F, K, T, D, is_call)

    positive = price > 0
    assert positive.sum() > 0.95 * price.size
    assert np.all(np.abs(implied[positive] - sigma[positive]) <= 1e-6)
    assert np.isnan(implied[~positive]).all()  # a zero price is on the lower bound


# ---------------------------------------------------------------------------
# Newton demo (DESIGN.md section 6)
# ---------------------------------------------------------------------------

def test_demo_price():
    assert round(_demo_price(), 8) == 0.05007364


def test_newton_naive_diverges():
    # From sigma0 = 0.20 vega is about 0.0013, so the first step overshoots to 38.24; there the
    # put is worth almost D·K and the next step gives a negative vol, where the run fails.
    iterates, converged = newton_naive(_demo_price(), **DEMO, sigma0=0.20)
    assert not converged
    assert iterates.size == 3
    assert iterates[0] == 0.20
    assert iterates[1] == pytest.approx(38.243449, abs=5e-7)
    assert iterates[2] == pytest.approx(-13054.718015, abs=5e-7)


def test_newton_manaster_koehler_converges():
    iterates, converged = newton_manaster_koehler(_demo_price(), **DEMO)
    assert converged
    assert iterates[0] == pytest.approx(2.911048, abs=5e-7)  # √(2|ln(F/K)|/T)
    assert iterates.size - 1 == 8  # Newton steps
    assert np.all(np.diff(iterates) < 0)  # monotone from above
    assert iterates[-1] == pytest.approx(DEMO_SIGMA, abs=1e-12)


def test_brent_on_demo():
    assert black76_implied_vol(_demo_price(), **DEMO) == pytest.approx(DEMO_SIGMA, abs=1e-12)


def test_newton_manaster_koehler_root_above_start():
    # A root above the start, where the price is concave in sigma: monotone from below (the last,
    # converged step can round to exactly 0).
    F, K, T, D = 100.0, 95.0, 0.25, np.exp(-0.01)
    price = float(black76_price(F, K, T, D, 0.9, False))
    iterates, converged = newton_manaster_koehler(price, F, K, T, D, is_call=False)
    assert converged and np.all(np.diff(iterates) >= 0)
    assert iterates[-1] == pytest.approx(0.9, abs=1e-12)


# ---------------------------------------------------------------------------
# No-arbitrage bounds and the bracket
# ---------------------------------------------------------------------------

def test_nan_outside_bounds():
    F, T, D = 100.0, 0.25, 0.99
    # Call K = 90: D·(F - K) = 9.9 < C < D·F = 99. Put K = 110: D·(K - F) = 9.9 < P < D·K = 108.9.
    for K, is_call, upper in [(90.0, True, 99.0), (110.0, False, 108.9)]:
        for price in (9.9, 9.0, upper, upper + 1.0):
            assert np.isnan(black76_implied_vol(price, F, K, T, D, is_call)), (K, price)
        assert np.isfinite(black76_implied_vol(9.95, F, K, T, D, is_call))
    # Out of the money the lower bound is 0: a zero or negative price has no implied vol.
    for K, is_call in [(110.0, True), (90.0, False)]:
        for price in (0.0, -0.01):
            assert np.isnan(black76_implied_vol(price, F, K, T, D, is_call))


def test_nan_without_sign_change():
    # Inside the bounds, but the implied vol lies outside [0.001, 5.0].
    F, K, T, D = 100.0, 100.0, 1.0, 0.96
    high = black76_price(F, K, T, D, 6.0)  # about 0.997·D·F, below the upper bound D·F
    low = black76_price(F, K, T, D, 0.0005)
    assert 0 < low < high < D * F
    assert np.isnan(black76_implied_vol(high, F, K, T, D))
    assert np.isnan(black76_implied_vol(low, F, K, T, D))
    for sigma in (0.0011, 4.99):  # just inside the bracket
        price = black76_price(F, K, T, D, sigma)
        assert black76_implied_vol(price, F, K, T, D) == pytest.approx(sigma, abs=1e-6)


def test_nan_for_non_finite_inputs():
    assert np.isnan(black76_implied_vol(np.nan, 100.0, 100.0, 1.0, 1.0))
    assert np.isnan(black76_implied_vol(5.0, np.inf, 100.0, 1.0, 1.0))
    assert np.isnan(black76_implied_vol(5.0, 100.0, 100.0, np.nan, 1.0))


def test_vectorised_matches_scalar():
    F, T, D = 100.0, 0.5, 0.98
    K = np.array([[90.0, 100.0, 110.0], [80.0, 105.0, 120.0]])
    is_call = K >= F
    price = black76_price(F, K, T, D, 0.25, is_call)
    price[0, 1] = 0.0  # no implied vol
    implied = black76_implied_vol(price, F, K, T, D, is_call)
    assert implied.shape == K.shape
    for i in np.ndindex(K.shape):
        scalar = black76_implied_vol(price[i], F, K[i], T, D, is_call[i])
        assert np.isnan(scalar) if i == (0, 1) else scalar == implied[i]
    assert np.isnan(implied[0, 1])
    np.testing.assert_allclose(np.delete(implied.ravel(), 1), 0.25, rtol=0, atol=1e-12)
    assert np.ndim(black76_implied_vol(price[0, 0], F, 90.0, T, D, False)) == 0


# ---------------------------------------------------------------------------
# Forward vega
# ---------------------------------------------------------------------------

def test_black76_vega_finite_difference():
    k_over_f, T, sigma = np.meshgrid(
        [0.8, 0.9, 1.0, 1.1, 1.25], [7 / 365, 0.25, 1.0, 2.0], [0.1, 0.2, 0.4], indexing="ij"
    )
    F, D = 100.0, np.exp(-0.04 * T)
    K, h = k_over_f * F, 1e-4
    for is_call in (True, False):
        def price(u):
            return black76_price(F, K, T, D, sigma + u, is_call)

        fd = (-price(2 * h) + 8 * price(h) - 8 * price(-h) + price(-2 * h)) / (12 * h)
        np.testing.assert_allclose(black76_vega(F, K, T, D, sigma), fd, rtol=1e-8, atol=1e-9)


def test_black76_vega_matches_spot_vega():
    S, r, q = 100.0, 0.04, 0.013
    K, T, sigma = np.meshgrid([80.0, 100.0, 125.0], [0.1, 1.0, 3.0], [0.15, 0.5], indexing="ij")
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    np.testing.assert_allclose(black76_vega(F, K, T, D, sigma), bs_vega(S, K, T, r, q, sigma), rtol=1e-12, atol=0)
