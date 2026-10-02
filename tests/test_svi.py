import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from volsurf.black_scholes import black76_vega
from volsurf.implied_vol import black76_implied_vol
from volsurf.svi import (
    SVIFit,
    fit_svi,
    fit_svi_direct,
    outer_starts,
    quasi_explicit_inner,
    raw_svi,
    svi_constraints,
    svi_fit_errors,
    svi_objective,
    vega_weights,
)

FROZEN_CHAIN = pathlib.Path(__file__).resolve().parents[1] / "data" / "frozen" / "chain_20260930.parquet"
FROZEN_CHAINS = ["chain_20260930.parquet", "chain_20261001.parquet", "chain_20261002.parquet"]  # every snapshot
NEAR_TARGET = 1.0  # DESIGN section 7: near-the-money RMSE under 1 vol point on every slice
CONSTRAINT_TOL = 1e-12  # DESIGN section 7: constraints hold to 1e-12 after the conversion to raw parameters
BOUND_TOL = 1e-8  # DESIGN section 7: m on its bound, or |rho| on 1, to this tolerance
CROSS_CHECK_TOL = 1e-10  # DESIGN section 7: the direct fit may not beat the quasi-explicit objective by more
SEED_NOISE = 20261003  # noise of the synthetic noisy slice
SEED_FEASIBLE = 20261004  # random feasible points of the inner problem

T = 0.25
K_GRID = np.linspace(-0.5, 0.25, 61)
TRUE = (0.002, 0.08, -0.55, 0.03, 0.12)  # a, b, rho, m, s: a skewed slice inside every constraint
# Lee's bound broken: b·(1 + |rho|) = 1.6·1.5625 = 2.5, so the put wing of w rises with slope 2.5.
LEE_BROKEN, LEE_T, LEE_K = (0.0, 1.6, -0.5625, 0.0, 0.1), 1.0, np.linspace(-0.5, 0.5, 41)


def _slice(params, k=K_GRID, T=T, noise=0.0):
    """Quotes (k, w, weights) of a raw SVI slice, with relative noise on the implied vol when noise > 0."""
    w = raw_svi(k, *params)
    iv = np.sqrt(w / T)
    if noise:
        iv = iv * (1.0 + noise * np.random.default_rng(SEED_NOISE).standard_normal(k.size))
        w = iv**2 * T
    return k, w, vega_weights(100.0, 100.0 * np.exp(k), T, 0.99, iv)


def _assert_feasible(fit):
    """Every raw SVI constraint holds to 1e-12 (|rho| <= 1 included), with s > 0 strictly; the flags match."""
    margins = svi_constraints(*fit[:5])
    assert min(margins.values()) >= -CONSTRAINT_TOL, margins
    assert margins["s"] > 0, margins
    assert fit.rho_on_bound == (margins["rho"] <= BOUND_TOL)


def _inner_objective(k, w, weights, m, s, a, c, d):
    y = (k - m) / s
    r = a + d * y + c * np.sqrt(y * y + 1.0) - w
    return float(weights @ (r * r) / weights.sum())


def test_raw_svi_values():
    a, b, rho, m, s = 0.01, 0.1, -0.5, 0.1, 0.2
    assert raw_svi(0.1, a, b, rho, m, s) == pytest.approx(0.01 + 0.1 * 0.2, abs=1e-15)  # k = m: w = a + b·s
    assert raw_svi(-0.2, a, b, rho, m, s) == pytest.approx(0.01 + 0.1 * (0.15 + math.sqrt(0.13)), abs=1e-15)
    assert raw_svi(0.5, a, b, rho, m, s) == pytest.approx(0.01 + 0.1 * (-0.2 + math.sqrt(0.2)), abs=1e-15)
    assert raw_svi(np.zeros((2, 3)), a, b, rho, m, s).shape == (2, 3)
    two_slopes = raw_svi(0.0, a, np.array([0.1, 0.2]), rho, m, s)  # parameters broadcast too
    np.testing.assert_allclose(two_slopes, a + np.array([0.1, 0.2]) * (0.05 + math.sqrt(0.05)), rtol=0, atol=1e-15)
    # The wings grow with slopes b·(1 + rho) on the right and b·(1 - rho) on the left.
    assert raw_svi(1e4 + 1, a, b, rho, m, s) - raw_svi(1e4, a, b, rho, m, s) == pytest.approx(0.05, abs=1e-9)
    assert raw_svi(-1e4 - 1, a, b, rho, m, s) - raw_svi(-1e4, a, b, rho, m, s) == pytest.approx(0.15, abs=1e-9)


def test_svi_constraints():
    margins = svi_constraints(0.01, 0.1, -0.5, 0.1, 0.2)
    assert margins == pytest.approx(
        {"b": 0.1, "rho": 0.5, "s": 0.2, "w_min": 0.01 + 0.1 * 0.2 * math.sqrt(0.75), "lee": 2 - 0.1 * 1.5}, abs=1e-15
    )
    assert svi_constraints(*LEE_BROKEN)["lee"] == pytest.approx(-0.5, abs=1e-15)  # Lee's bound broken
    assert svi_constraints(-0.1, 0.1, 0.0, 0.0, 0.2)["w_min"] == pytest.approx(-0.08, abs=1e-15)  # w < 0 at m
    beyond = svi_constraints(0.01, 0.1, 1.2, 0.0, 0.2)
    assert beyond["rho"] == pytest.approx(-0.2, abs=1e-15) and beyond["w_min"] == 0.01
    assert svi_constraints(0.01, -0.1, 0.0, 0.0, 0.2)["b"] == -0.1
    assert svi_constraints(0.01, 0.1, 0.0, 0.0, 0.0)["s"] == 0.0  # s > 0 needs a positive margin
    array = svi_constraints(np.array([0.01, -0.1]), 0.1, 0.0, 0.0, 0.2)["w_min"]
    np.testing.assert_allclose(array, [0.03, -0.08], rtol=0, atol=1e-15)


def test_vega_weights():
    k = np.linspace(-0.3, 0.2, 11)
    K, iv = 100.0 * np.exp(k), 0.2 - 0.3 * k
    weights = vega_weights(100.0, K, T, 0.99, iv)
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)
    ratio = weights / black76_vega(100.0, K, T, 0.99, iv)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-13, atol=0)  # proportional to vega
    with pytest.raises(ValueError, match="vega weights"):
        vega_weights(100.0, K, T, 0.99, np.where(k > 0, np.nan, iv))


def test_svi_objective():
    k, w, weights = _slice(TRUE, noise=0.01)
    params = (0.003, 0.07, -0.5, 0.0, 0.1)
    expected = float(np.sum(weights * (raw_svi(k, *params) - w) ** 2))
    assert svi_objective(k, w, weights, *params) == pytest.approx(expected, rel=1e-14)
    assert svi_objective(k, raw_svi(k, *params), weights, *params) == 0.0


def test_inner_interior_is_the_closed_form():
    # Noiseless at the true (m, s): the inner problem gives back a, c = b·s and d = rho·b·s.
    a, b, rho, m, s = TRUE
    k, w, weights = _slice(TRUE)
    inner = quasi_explicit_inner(k, w, weights, m, s)
    np.testing.assert_allclose(inner[:3], [a, b * s, rho * b * s], rtol=0, atol=1e-12)
    assert inner[3] <= 1e-24

    # Noisy at another (m, s): the weighted least squares solution, feasible here, is the answer.
    k, w, weights = _slice(TRUE, noise=0.01)
    m, s = 0.0, 0.15
    y = (k - m) / s
    X = np.column_stack([np.ones_like(y), np.sqrt(y * y + 1.0), y])
    root = np.sqrt(weights)
    closed, *_ = np.linalg.lstsq(root[:, None] * X, root * w, rcond=None)
    a0, c0, d0 = closed
    assert c0 >= abs(d0) and c0 + abs(d0) <= 2 * s and a0 + math.sqrt(c0 * c0 - d0 * d0) >= 0  # interior
    inner = quasi_explicit_inner(k, w, weights, m, s)
    np.testing.assert_allclose(inner[:3], closed, rtol=1e-10, atol=0)
    assert inner[3] == pytest.approx(_inner_objective(k, w, weights, m, s, *closed), rel=1e-12)


def test_inner_constrained_by_lee():
    k, w, weights = _slice(LEE_BROKEN, k=LEE_K, T=LEE_T)
    m, s = 0.0, 0.1
    y = (k - m) / s
    X = np.column_stack([np.ones_like(y), np.sqrt(y * y + 1.0), y])
    root = np.sqrt(weights)
    (_, c0, d0), *_ = np.linalg.lstsq(root[:, None] * X, root * w, rcond=None)
    assert c0 + abs(d0) > 2 * s  # the unconstrained solution breaks Lee's bound

    a, c, d, objective = quasi_explicit_inner(k, w, weights, m, s)
    assert c >= abs(d) and c + abs(d) <= 2 * s + CONSTRAINT_TOL and a + math.sqrt(c * c - d * d) >= 0
    assert c + abs(d) >= 2 * s - 1e-9  # Lee's bound is active
    assert objective == pytest.approx(_inner_objective(k, w, weights, m, s, a, c, d), rel=1e-12)

    # No random feasible point does better.
    rng = np.random.default_rng(SEED_FEASIBLE)
    c_r = rng.uniform(0.0, 2 * s, 2000)
    d_max = np.minimum(c_r, 2 * s - c_r)
    d_r = rng.uniform(-1.0, 1.0, 2000) * d_max
    floor = -np.sqrt(c_r**2 - d_r**2)
    a_r = floor + rng.uniform(0.0, 1.0, 2000) * (w.max() - floor)
    others = [_inner_objective(k, w, weights, m, s, *p) for p in zip(a_r, c_r, d_r)]
    assert objective <= min(others)


def test_fit_svi_recovers_noiseless_slice():
    # DESIGN section 7 acceptance: a noiseless synthetic slice is recovered to 1e-8 in w.
    k, w, weights = _slice(TRUE)
    fit = fit_svi(k, w, weights)
    assert isinstance(fit, SVIFit) and not fit.m_on_bound and not fit.rho_on_bound
    grid = np.linspace(k.min(), k.max(), 2001)
    error = max(np.abs(raw_svi(k, *fit[:5]) - w).max(), np.abs(raw_svi(grid, *fit[:5]) - raw_svi(grid, *TRUE)).max())
    print(f"noiseless slice: largest error in w {error:.2e}")
    assert error <= 1e-8
    _assert_feasible(fit)
    assert fit.objective == pytest.approx(svi_objective(k, w, weights, *fit[:5]), abs=1e-20)


def test_fit_svi_respects_lee_bound():
    k, w, weights = _slice(LEE_BROKEN, k=LEE_K, T=LEE_T)
    fit = fit_svi(k, w, weights)
    _assert_feasible(fit)
    lee = fit.b * (1 + abs(fit.rho))
    print(f"Lee-broken slice: fitted b·(1 + |rho|) = {lee:.12f}")
    assert 2 - 1e-6 <= lee <= 2 + CONSTRAINT_TOL  # the bound holds and is active
    assert k.min() <= fit.m <= k.max()


def test_fit_svi_bounds_m_to_the_quoted_range():
    # The vertex m = 0.4 lies beyond the largest quoted k = 0.25: the fit stops m on that bound. With
    # the vertex held back, the closest smile flattens its right wing completely, rho = -1, which the
    # closed set |d| <= c allows and the fit flags.
    beyond = (0.002, 0.08, -0.55, 0.4, 0.12)
    k, w, weights = _slice(beyond)
    fit = fit_svi(k, w, weights)
    assert fit.m == pytest.approx(k.max(), abs=BOUND_TOL) and fit.m_on_bound
    assert fit.rho == pytest.approx(-1.0, abs=BOUND_TOL) and fit.rho_on_bound
    _assert_feasible(fit)


def test_direct_fit_does_not_beat_quasi_explicit():
    # DESIGN section 7 cross-check, on a noisy slice and on the Lee-broken slice, where a constraint binds.
    for k, w, weights in (_slice(TRUE, noise=0.01), _slice(LEE_BROKEN, k=LEE_K, T=LEE_T)):
        quasi, direct = fit_svi(k, w, weights), fit_svi_direct(k, w, weights)
        print(f"quasi-explicit {quasi.objective:.6e}, direct {direct.objective:.6e}, "
              f"gap {direct.objective - quasi.objective:.2e}")
        assert direct.objective >= quasi.objective - CROSS_CHECK_TOL
        _assert_feasible(quasi)
        _assert_feasible(direct)
        assert direct.objective == pytest.approx(svi_objective(k, w, weights, *direct[:5]), abs=1e-20)
    # Its starts come from the documented seed: the same seed gives the same fit.
    assert fit_svi_direct(k, w, weights) == direct


def test_outer_starts():
    # Nine starts, m0 in {-1, 0, 1}·h clipped into the quoted range, then s0 in {0.5, 1, 2}·h, h = √(Σ ω·w).
    k, w, weights = _slice(TRUE)
    h = math.sqrt((weights / weights.sum()) @ w)
    starts = outer_starts(k, w, weights)
    assert len(starts) == 9
    expected = [(min(max(m0 * h, k.min()), k.max()), s0 * h) for m0 in (-1, 0, 1) for s0 in (0.5, 1, 2)]
    np.testing.assert_allclose(starts, expected, rtol=1e-15, atol=0)
    # A narrow quoted range clips m0 onto its ends.
    narrow = k[(k >= -0.05) & (k <= 0.05)]  # 9 quotes, narrower than h
    starts = outer_starts(narrow, raw_svi(narrow, *TRUE), np.ones(narrow.size))
    assert {m0 for m0, _ in starts} == {narrow.min(), 0.0, narrow.max()}


def test_fit_svi_rejects_bad_slices():
    k, w, weights = _slice(TRUE)
    with pytest.raises(ValueError, match="at least 5 quotes"):
        fit_svi(k[:4], w[:4], weights[:4])
    with pytest.raises(ValueError, match="same length"):
        fit_svi(k, w[:-1], weights)
    with pytest.raises(ValueError, match="positive"):
        fit_svi(k, np.where(k > 0, 0.0, w), weights)
    with pytest.raises(ValueError, match="finite"):
        fit_svi_direct(k, np.where(k > 0, np.nan, w), weights)
    with pytest.raises(ValueError, match="weights"):
        quasi_explicit_inner(k, w, -weights, 0.0, 0.1)


def test_svi_fit_errors():
    # A flat fit at 20% vol against market vols of 25, 22, 21 and 19%: errors of -5, -2, -1 and +1 vol
    # points. w_ATM = 0.21², so the band is |k| <= 0.21, which leaves out k = -0.3.
    k = np.array([-0.3, -0.1, 0.0, 0.1])
    w = np.array([0.25, 0.22, 0.21, 0.19]) ** 2
    flat = SVIFit(0.04, 0.0, 0.0, 0.0, 0.1, 0.0, False, False)
    errors = svi_fit_errors(k, w, 1.0, flat)
    assert errors.band == pytest.approx(0.21, abs=1e-15) and errors.n_near == 3
    assert errors.rmse_near == pytest.approx(math.sqrt((4 + 1 + 1) / 3), abs=1e-12)
    assert errors.rmse_all == pytest.approx(math.sqrt((25 + 4 + 1 + 1) / 4), abs=1e-12)
    # T scales the vols: the same total variances at T = 0.5 are vols √2 times higher, and so are the errors.
    half = svi_fit_errors(k, w, 0.5, flat)
    assert half.rmse_all == pytest.approx(errors.rmse_all * math.sqrt(2), abs=1e-12)
    with pytest.raises(ValueError, match="span k = 0"):
        svi_fit_errors(k[:2], w[:2], 1.0, flat)


@pytest.mark.parametrize("chain_name", FROZEN_CHAINS)
def test_frozen_svi_fits(chain_name):
    # DESIGN section 7 on every frozen snapshot: every slice of the frozen chain, fitted from its kept
    # quotes with an implied vol, satisfies the constraints with m inside its quoted range, passes the
    # direct cross-check and fits within 1 vol point RMSE near the money.
    chain = pd.read_parquet(FROZEN_CHAIN.with_name(chain_name))
    kept = chain[chain["status"] == "kept"].copy()
    kept["iv"] = black76_implied_vol(
        kept["mid"], kept["F"], kept["strike"], kept["T"], kept["D"], kept["option_type"] == "call"
    )
    kept = kept.dropna(subset=["iv"])
    kept["w"] = kept["iv"] ** 2 * kept["T"]
    slices = kept.groupby("expiry")
    assert len(slices) == 12
    for expiry, rows in slices:
        rows = rows.sort_values("k")  # as the notebook fits them
        k, w, T = rows["k"].to_numpy(), rows["w"].to_numpy(), rows["T"].iloc[0]
        weights = vega_weights(rows["F"], rows["strike"], rows["T"], rows["D"], rows["iv"])
        quasi, direct = fit_svi(k, w, weights), fit_svi_direct(k, w, weights)
        errors = svi_fit_errors(k, w, T, quasi)
        print(f"{expiry:%Y-%m-%d}: near {errors.rmse_near:.2f}, all {errors.rmse_all:.2f} vol points; "
              f"direct - quasi-explicit {direct.objective - quasi.objective:.1e}")
        _assert_feasible(quasi)
        assert k.min() <= quasi.m <= k.max()
        assert direct.objective >= quasi.objective - CROSS_CHECK_TOL
        assert errors.rmse_near < NEAR_TARGET
