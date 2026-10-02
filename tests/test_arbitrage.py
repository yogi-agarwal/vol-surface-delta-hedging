import math
import pathlib
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from volsurf.arbitrage import (
    N_CERTIFY,
    N_FINE,
    N_REFIT,
    VIOLATION_TOL,
    RefitResult,
    _g_and_jacobian,
    _inner_parts,
    _margin_minimisers,
    arbitrage_table,
    butterfly_g,
    k_grid,
    refit_restarts,
    refit_slice,
    refit_surface,
    svi_density,
    svi_derivatives,
    violation_counts,
)
from volsurf.black_scholes import black76_price
from volsurf.implied_vol import black76_implied_vol
from volsurf.svi import fit_svi, raw_svi, svi_constraints, svi_fit_errors, svi_objective, vega_weights

FROZEN_CHAIN = pathlib.Path(__file__).resolve().parents[1] / "data" / "frozen" / "chain_20260930.parquet"
NEAR_TARGET = 1.0  # DESIGN section 8: near-the-money RMSE under 1 vol point on every slice after the refit
CONSTRAINT_TOL = 1e-12  # DESIGN section 7: Stage 3 constraints after the conversion to raw parameters
CROSS_CHECK_TOL = 1e-10  # DESIGN section 8: no restart may beat the refit by more on f

# DESIGN section 8 test slice: a, b, rho, m, s at T = 30/365.
DESIGN_SLICE = (0.0010, 0.012, -0.75, 0.015, 0.03)
DESIGN_T = 30 / 365
# DESIGN section 8 (verified): the density at six points, as printed there.
DESIGN_DENSITY = {-0.3: "0.00483804", -0.2: "0.0590773", -0.1: "0.782553", 0.0: "11.2865", 0.1: "0.196568",
                  0.2: "2.00322e-05"}
LEE_EDGE = (0.0, 1.0, 0.0, 0.0, 0.01)  # w = √(k² + 0.0001): wings of slope 1 on a level near 0


def _numerical_derivatives(k, params, h=1e-5):
    """w' and w'' of raw SVI by central differences, independent of svi_derivatives."""
    up, mid, down = (raw_svi(k + d, *params) for d in (h, 0.0, -h))
    return (up - down) / (2 * h), (up - 2 * mid + down) / (h * h)


def test_k_grid():
    for n, step in ((N_REFIT, 0.015), (N_FINE, 0.0005), (N_CERTIFY, 0.00005)):
        k = k_grid(n)
        assert k.size == n and k[0] == -2.0 and k[-1] == 1.0
        np.testing.assert_allclose(np.diff(k), step, rtol=1e-9, atol=0)
    # The certification grid nests the fine grid, which nests the refit grid (every tenth and thirtieth point).
    np.testing.assert_allclose(k_grid(N_CERTIFY)[::10], k_grid(N_FINE), rtol=0, atol=1e-15)
    np.testing.assert_allclose(k_grid(N_FINE)[::30], k_grid(N_REFIT), rtol=0, atol=1e-15)


def test_svi_derivatives():
    k = np.linspace(-0.4, 0.3, 15)
    w, dw, d2w = svi_derivatives(k, *DESIGN_SLICE)
    np.testing.assert_allclose(w, raw_svi(k, *DESIGN_SLICE), rtol=1e-15, atol=0)
    num_dw, num_d2w = _numerical_derivatives(k, DESIGN_SLICE)
    # Central differences err by about h²·w'''/6, up to 2e-10 near the vertex, where w''' ~ b/s².
    np.testing.assert_allclose(dw, num_dw, rtol=0, atol=1e-9)
    np.testing.assert_allclose(d2w, num_d2w, rtol=1e-4, atol=1e-6)
    # At the vertex k = m: w' = b·rho and w'' = b/s.
    a, b, rho, m, s = DESIGN_SLICE
    _, dw_m, d2w_m = svi_derivatives(m, *DESIGN_SLICE)
    assert dw_m == pytest.approx(b * rho, rel=1e-15) and d2w_m == pytest.approx(b / s, rel=1e-14)


def test_butterfly_g():
    # A flat smile, w constant, has g = 1.
    np.testing.assert_array_equal(butterfly_g(np.linspace(-2, 1, 7), 0.04, 0.0, 0.0, 0.0, 0.1), 1.0)
    # The formula, evaluated on numerical derivatives of w.
    k = np.linspace(-0.4, 0.3, 15)
    w = raw_svi(k, *DESIGN_SLICE)
    dw, d2w = _numerical_derivatives(k, DESIGN_SLICE)
    expected = (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2
    np.testing.assert_allclose(butterfly_g(k, *DESIGN_SLICE), expected, rtol=1e-4, atol=1e-6)
    # Slope-1 wings on a level near 0: for |k| well above s, g ≈ 3/16 - 1/(4|k|), negative below |k| = 4/3.
    assert butterfly_g(0.5, *LEE_EDGE) < 0 and butterfly_g(-0.5, *LEE_EDGE) < 0
    assert butterfly_g(1.9, *LEE_EDGE) > 0 and butterfly_g(-1.9, *LEE_EDGE) > 0
    assert butterfly_g(0.5, *LEE_EDGE) == pytest.approx(3 / 16 - 1 / 2, abs=1e-3)


def test_svi_density_design_values():
    # DESIGN section 8 (verified): each value agrees with the printed digits, to half a unit in the last.
    for k, printed in DESIGN_DENSITY.items():
        half_unit = 0.5 * 10.0 ** Decimal(printed).as_tuple().exponent
        p = float(svi_density(k, *DESIGN_SLICE))
        print(f"k = {k:+.1f}: p = {p:.9g} against {printed}")
        assert abs(p - float(printed)) <= half_unit


def test_svi_density_is_breeden_litzenberger():
    # K·∂²C/∂K² of the forward call per unit of F, by central differences of Black-76 prices.
    def call(K):
        w = raw_svi(np.log(K), *DESIGN_SLICE)
        return black76_price(1.0, K, 1.0, 1.0, np.sqrt(w), is_call=True)

    for k in DESIGN_DENSITY:
        K, h = math.exp(k), 1e-4
        second = (call(K + h) - 2 * call(K) + call(K - h)) / (h * h)
        assert float(svi_density(k, *DESIGN_SLICE)) == pytest.approx(K * second, rel=1e-4)


def test_svi_density_integrates_to_one():
    # A density of k = ln(S_T/F): total mass 1 and E[S_T/F] = ∫ e^k p(k) dk = 1.
    def density(k):
        return float(svi_density(k, *DESIGN_SLICE))

    mass = sum(quad(density, lo, hi, limit=200, epsabs=1e-13)[0] for lo, hi in ((-3.0, 0.0), (0.0, 2.0)))
    mean = sum(quad(lambda k: math.exp(k) * density(k), lo, hi, limit=200, epsabs=1e-13)[0]
               for lo, hi in ((-3.0, 0.0), (0.0, 2.0)))
    assert mass == pytest.approx(1.0, abs=1e-8) and mean == pytest.approx(1.0, abs=1e-8)
    # Where g < 0 the density is negative.
    assert svi_density(0.5, *LEE_EDGE) < 0


def test_violation_counts():
    k = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    margin = np.array([-1.0, -1e-13, -1e-11, np.nan, 2.0])
    # -1e-13 is within the tolerance, -1e-11 is not, and NaN counts as a violation.
    assert violation_counts(margin, k, -0.6, 0.2) == (1, 2)
    assert violation_counts(margin, k, -2.0, 2.0) == (3, 0)
    assert violation_counts(margin, k, -0.6, 0.2, tol=0.0) == (2, 2)
    with pytest.raises(ValueError, match="same shape"):
        violation_counts(margin[:-1], k, -0.6, 0.2)


def test_arbitrage_table():
    k = k_grid(N_FINE)
    flat = (0.04, 0.0, 0.0, 0.0, 0.1)  # w = 0.04
    # w = 0.03 + 0.05·√(k² + 0.01) is below 0.04 for |k| < √0.03 = 0.1732...
    crossing = (0.03, 0.05, 0.0, 0.0, 0.1)
    ranges = [(-0.5, 0.5), (-0.10025, 0.30025)]  # bounds between grid points
    table = arbitrage_table([flat, crossing, LEE_EDGE], [*ranges, (-1.0, 0.8)], k, index=["A", "B", "C"])
    assert list(table.index) == ["A", "B", "C"]
    assert table.loc["A", "butterfly quoted"] == 0 and table.loc["A", "min g"] == 1.0
    assert table.loc[["A"], ["calendar quoted", "calendar extrapolated", "min calendar margin"]].isna().all(axis=None)
    # B against A: violations for |k| < 0.1732; quoted where both are quoted, [-0.10025, 0.30025].
    below = np.abs(k) < math.sqrt(0.03)
    assert below.sum() == 693
    assert (table.loc["B", "calendar quoted"], table.loc["B", "calendar extrapolated"]) == (547, 146)
    assert table.loc["B", "min calendar margin"] == pytest.approx(0.03 + 0.05 * 0.1 - 0.04, abs=1e-15)
    assert table.loc["B", "butterfly quoted"] == table.loc["B", "butterfly extrapolated"] == 0
    # C has negative g for |k| below about 4/3, counted in its own range [-1, 0.8], and falls below B.
    g = butterfly_g(k, *LEE_EDGE)
    assert table.loc["C", "butterfly quoted"] == ((g < 0) & (k >= -1.0) & (k <= 0.8)).sum() > 0
    assert table.loc["C", "butterfly extrapolated"] == ((g < 0) & ((k < -1.0) | (k > 0.8))).sum() > 0
    assert table.loc["C", "min g"] == pytest.approx(g.min(), rel=1e-15)
    assert table.loc["C", "calendar quoted"] > 0
    assert str(table["butterfly quoted"].dtype) == "Int64"
    with pytest.raises(ValueError, match="one entry per slice"):
        arbitrage_table([flat], ranges, k)


# Refit test slices. A 16-day slice whose left wing breaks g in the extrapolated region (the Stage 3 fit of
# 2026-10-16, rounded), quoted on [-0.316, 0.077]; and a 60-day slice whose flatter left wing falls below the
# DESIGN slice for k below about -0.33, quoted on [-0.3, 0.2].
SHORT, SHORT_T, SHORT_K = (-0.026, 0.0839, 0.12, 0.0734, 0.3185), 16 / 365, np.linspace(-0.316, 0.077, 80)
LATER, LATER_T, LATER_K = (0.002, 0.012, -0.5, 0.015, 0.05), 60 / 365, np.linspace(-0.3, 0.2, 60)


def _quotes(params, k, T):
    """Noiseless quotes (k, w, weights) of a raw SVI slice, vega-weighted as in Stage 3."""
    w = raw_svi(k, *params)
    return k, w, vega_weights(1.0, np.exp(k), T, 1.0, np.sqrt(w / T))


def _assert_clean(fit, k_quotes, previous=None):
    """No violation on the certification grid, Stage 3 constraints to 1e-12 with s > 0, m in the quoted range."""
    k = k_grid(N_CERTIFY)
    assert butterfly_g(k, *fit[:5]).min() >= -VIOLATION_TOL
    if previous is not None:
        assert (raw_svi(k, *fit[:5]) - raw_svi(k, *previous)).min() >= -VIOLATION_TOL
    margins = svi_constraints(*fit[:5])
    assert min(margins.values()) >= -CONSTRAINT_TOL and margins["s"] > 0, margins
    assert k_quotes.min() <= fit.m <= k_quotes.max()


def test_inner_parts_and_g_jacobian():
    # The analytic gradients in (a, c, d, m, s) against central differences.
    k = np.linspace(-1.5, 0.8, 40)
    for a, b, rho, m, s in (DESIGN_SLICE, SHORT, LATER):
        p = np.array([a, b * s, rho * b * s, m, s])
        w, w1, w2, dw, dw1, dw2 = _inner_parts(k, p)
        np.testing.assert_allclose(w, raw_svi(k, a, b, rho, m, s), rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(np.stack([w1, w2]), np.stack(svi_derivatives(k, a, b, rho, m, s)[1:]),
                                   rtol=1e-12, atol=1e-14)
        g, jac = _g_and_jacobian(k, p)
        np.testing.assert_allclose(g, butterfly_g(k, a, b, rho, m, s), rtol=1e-12, atol=1e-12)
        for i in range(5):
            step = np.zeros(5)
            step[i] = 1e-6 * max(abs(p[i]), 1e-3)
            up, down = _inner_parts(k, p + step), _inner_parts(k, p - step)
            for value, grad in ((0, dw), (1, dw1), (2, dw2)):
                numeric = (up[value] - down[value]) / (2 * step[i])
                np.testing.assert_allclose(grad[:, i], numeric, rtol=1e-5, atol=1e-7 * np.abs(numeric).max())
            numeric = (_g_and_jacobian(k, p + step)[0] - _g_and_jacobian(k, p - step)[0]) / (2 * step[i])
            np.testing.assert_allclose(jac[:, i], numeric, rtol=1e-5, atol=1e-7 * np.abs(numeric).max())


def test_margin_minimisers_catch_dips_between_grid_points():
    # A V-shaped smile with its vertex midway between the fine-grid points 0 and 0.0005: against a flat
    # slice at 0.04, the calendar margin is positive at every grid point but -5e-5 at the vertex.
    fine = k_grid(N_FINE)
    flat, vee = (0.04, 0.0, 0.0, 0.0, 0.1), (0.04 - 1.5e-4, 1.0, 0.0, 0.00025, 1e-4)
    assert (raw_svi(fine, *vee) - raw_svi(fine, *flat)).min() > 0
    found = _margin_minimisers(fine, vee, flat, VIOLATION_TOL)
    vertex = found[np.abs(found - 0.00025) < 1e-6]
    assert vertex.size == 1 and vertex[0] == pytest.approx(0.00025, abs=1e-9)
    assert raw_svi(vertex[0], *vee) - raw_svi(vertex[0], *flat) == pytest.approx(-5e-5, rel=1e-6)
    # Without a dip below the tolerance there is nothing to add.
    assert _margin_minimisers(fine, flat, None, VIOLATION_TOL).size == 0


def test_refit_keeps_a_feasible_start():
    k, w, weights = _quotes(DESIGN_SLICE, np.linspace(-0.3, 0.2, 50), DESIGN_T)
    start = fit_svi(k, w, weights)
    result = refit_slice(k, w, weights, start)
    assert isinstance(result, RefitResult) and not result.refitted and result.rounds == 0
    assert result.fit is start and result.added.size == 0
    # A start given as raw parameters comes back as an SVIFit with its objective.
    result = refit_slice(k, w, weights, DESIGN_SLICE)
    assert not result.refitted and tuple(result.fit[:5]) == DESIGN_SLICE
    assert result.fit.objective == pytest.approx(0.0, abs=1e-30)


def test_refit_removes_butterfly_arbitrage():
    k, w, weights = _quotes(SHORT, SHORT_K, SHORT_T)
    assert butterfly_g(k_grid(N_FINE), *SHORT).min() < -0.02  # the left wing breaks g
    result = refit_slice(k, w, weights, SHORT)
    assert result.refitted and result.added.size > 0
    _assert_clean(result.fit, k)
    assert result.fit.objective == pytest.approx(svi_objective(k, w, weights / weights.sum(), *result.fit[:5]),
                                                 rel=1e-12)
    errors = svi_fit_errors(k, w, SHORT_T, result.fit)
    print(f"short slice refit: near-the-money RMSE {errors.rmse_near:.3f} vol points, {result.added.size} points added")
    assert errors.rmse_near < NEAR_TARGET


def test_refit_removes_calendar_arbitrage():
    k, w, weights = _quotes(LATER, LATER_K, LATER_T)
    fine = k_grid(N_FINE)
    assert (raw_svi(fine, *LATER) - raw_svi(fine, *DESIGN_SLICE)).min() < -0.005  # below the DESIGN slice
    assert butterfly_g(fine, *LATER).min() > 0
    assert not refit_slice(k, w, weights, LATER).refitted  # without the earlier slice it is kept
    result = refit_slice(k, w, weights, LATER, previous=DESIGN_SLICE)
    assert result.refitted
    _assert_clean(result.fit, k, previous=DESIGN_SLICE)
    assert svi_fit_errors(k, w, LATER_T, result.fit).rmse_near < NEAR_TARGET


def test_refit_surface_and_restarts():
    quotes = [_quotes(DESIGN_SLICE, np.linspace(-0.3, 0.2, 50), DESIGN_T), _quotes(LATER, LATER_K, LATER_T)]
    results = refit_surface(quotes, [DESIGN_SLICE, LATER])
    assert len(results) == 2 and not results[0].refitted and results[1].refitted
    # The later slice is refitted against the earlier slice's refit, as refit_slice does it.
    alone = refit_slice(*quotes[1], LATER, previous=results[0].fit[:5])
    assert results[1].fit == alone.fit
    _assert_clean(results[1].fit, quotes[1][0], previous=results[0].fit[:5])
    # The nine restarts from the Stage 3 outer starts do not beat it by more than 1e-10 on f.
    restarts = refit_restarts(*quotes[1], previous=results[0].fit[:5])
    assert len(restarts) == 9
    solved = [r for r in restarts if r is not None]
    assert solved and all(r.refitted for r in solved)
    assert min(r.fit.objective for r in solved) >= results[1].fit.objective - CROSS_CHECK_TOL
    with pytest.raises(ValueError, match="one entry per slice"):
        refit_surface(quotes, [DESIGN_SLICE])


@pytest.fixture(scope="module")
def frozen_surface():
    """The twelve 2026-09-30 slices: quotes, quoted ranges, T, Stage 3 fits and the constrained refit."""
    chain = pd.read_parquet(FROZEN_CHAIN)
    kept = chain[chain["status"] == "kept"].copy()
    kept["iv"] = black76_implied_vol(
        kept["mid"], kept["F"], kept["strike"], kept["T"], kept["D"], kept["option_type"] == "call"
    )
    kept = kept.dropna(subset=["iv"])
    kept["w"] = kept["iv"] ** 2 * kept["T"]
    expiries, quotes, T, fits = [], [], [], []
    for expiry, rows in kept.groupby("expiry"):
        rows = rows.sort_values("k")  # as the notebook fits them
        k, w = rows["k"].to_numpy(), rows["w"].to_numpy()
        weights = vega_weights(rows["F"], rows["strike"], rows["T"], rows["D"], rows["iv"])
        expiries.append(expiry)
        quotes.append((k, w, weights))
        T.append(rows["T"].iloc[0])
        fits.append(fit_svi(k, w, weights))
    return {"expiries": expiries, "quotes": quotes, "T": np.array(T), "fits": fits,
            "refits": refit_surface(quotes, fits)}


def test_frozen_refit(frozen_surface):
    # DESIGN section 8 acceptance on the 2026-09-30 snapshot: after the refit, no violation on the
    # certification grid in either region, the Stage 3 constraints, m in the quoted range, and
    # near-the-money RMSE under 1 vol point on every slice.
    quotes, refits = frozen_surface["quotes"], frozen_surface["refits"]
    assert len(refits) == 12
    params = [r.fit[:5] for r in refits]
    ranges = [(k.min(), k.max()) for k, _, _ in quotes]
    table = arbitrage_table(params, ranges, k_grid(N_CERTIFY))
    counts = ["butterfly quoted", "butterfly extrapolated", "calendar quoted", "calendar extrapolated"]
    print(table.to_string())
    assert (table[counts].fillna(0) == 0).all(axis=None)
    cert = k_grid(N_CERTIFY)
    for expiry, (k, w, weights), T, result, previous in zip(
        frozen_surface["expiries"], quotes, frozen_surface["T"], refits, [None, *params[:-1]]
    ):
        _assert_clean(result.fit, k, previous)
        errors = svi_fit_errors(k, w, T, result.fit)
        print(f"{expiry:%Y-%m-%d}: refitted {result.refitted}, {result.added.size} points added, "
              f"near-the-money RMSE {errors.rmse_near:.3f} vol points")
        assert errors.rmse_near < NEAR_TARGET
        # Constraint points come from the fine grid or from searches between its points, never from the
        # certification points that the fine grid does not share.
        on_fine = np.isin(result.added, k_grid(N_FINE))
        assert not np.isin(result.added[~on_fine], cert).any()


def test_frozen_refit_restarts(frozen_surface):
    # DESIGN section 8 cross-check: the refit repeated from the nine Stage 3 outer starts never beats the
    # chosen refit by more than 1e-10 on f.
    previous = None
    for (k, w, weights), result in zip(frozen_surface["quotes"], frozen_surface["refits"]):
        solved = [r for r in refit_restarts(k, w, weights, previous) if r is not None]
        assert solved
        assert min(r.fit.objective for r in solved) >= result.fit.objective - CROSS_CHECK_TOL
        previous = result.fit[:5]
