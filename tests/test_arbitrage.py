import math
import pathlib
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad, simpson

import volsurf.arbitrage
from volsurf.arbitrage import (
    N_CERTIFY,
    N_FINE,
    N_REFIT,
    T_BRIDGE,
    VIOLATION_TOL,
    RefitResult,
    _g_and_jacobian,
    _inner_parts,
    _margin_minimisers,
    arbitrage_table,
    bracket,
    bridge_30d,
    butterfly_g,
    interpolate_w,
    k_grid,
    otm_per_strike,
    refit_restarts,
    refit_slice,
    refit_surface,
    svi_density,
    surface_from_chain,
    svi_derivatives,
    variance_swap_replication,
    variance_swap_z_integral,
    violation_counts,
    vix_cell_range,
    vix_discrete_variance,
)
from volsurf.black_scholes import black76_price
from volsurf.implied_vol import black76_implied_vol
from volsurf.svi import fit_svi, raw_svi, svi_constraints, svi_fit_errors, svi_objective, vega_weights

FROZEN_CHAIN = pathlib.Path(__file__).resolve().parents[1] / "data" / "frozen" / "chain_20260930.parquet"
FROZEN_CHAINS = ["chain_20260930.parquet", "chain_20261001.parquet", "chain_20261002.parquet"]  # every snapshot
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


def test_refit_keeps_only_a_feasible_line_search_stop(monkeypatch):
    # Every SLSQP run is made to report failure. A stop on a failed line search (exit mode 8) at a point that
    # meets every constraint is kept; another exit mode, or the same stop at an infeasible point, raises.
    k, w, weights = _quotes(SHORT, SHORT_K, SHORT_T)
    expected = refit_slice(k, w, weights, SHORT)
    real_minimize = volsurf.arbitrage.minimize

    def failing(status, spoil=None):
        def fake(*args, **kwargs):
            result = real_minimize(*args, **kwargs)
            if spoil is not None:
                result.x = spoil(result.x.copy())
            result.success, result.status, result.message = False, status, f"exit mode {status}"
            return result
        return fake

    def break_c(x):
        x[1] = -abs(x[2]) - 1.0  # c below -|d|: breaks c >= |d|
        return x

    monkeypatch.setattr(volsurf.arbitrage, "minimize", failing(8))
    kept = refit_slice(k, w, weights, SHORT)
    assert kept.refitted
    _assert_clean(kept.fit, k)
    assert kept.fit.objective == pytest.approx(expected.fit.objective, rel=1e-8)
    monkeypatch.setattr(volsurf.arbitrage, "minimize", failing(4))
    with pytest.raises(RuntimeError, match="did not converge: exit mode 4"):
        refit_slice(k, w, weights, SHORT)
    monkeypatch.setattr(volsurf.arbitrage, "minimize", failing(8, break_c))
    with pytest.raises(RuntimeError, match="did not converge: exit mode 8"):
        refit_slice(k, w, weights, SHORT)


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


@pytest.fixture(scope="module", params=FROZEN_CHAINS)
def snapshot_surface(request):
    """The surface of each frozen snapshot, from surface_from_chain: (chain file name, Surface)."""
    return request.param, surface_from_chain(pd.read_parquet(FROZEN_CHAIN.with_name(request.param)))


def test_surface_from_chain_matches_the_notebook_path(frozen_surface):
    # On 2026-09-30, the pipeline function reproduces the steps of the notebook's surface sections, which
    # the frozen_surface fixture follows: the same quotes, Stage 3 fits and refits.
    surface = surface_from_chain(pd.read_parquet(FROZEN_CHAIN))
    assert surface.expiries == frozen_surface["expiries"]
    np.testing.assert_array_equal(surface.T, frozen_surface["T"])
    for mine, theirs in zip(surface.quotes, frozen_surface["quotes"]):
        for a, b in zip(mine, theirs):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert surface.ranges == [(k.min(), k.max()) for k, _, _ in frozen_surface["quotes"]]
    assert surface.fits == frozen_surface["fits"]
    assert [r.fit for r in surface.refits] == [r.fit for r in frozen_surface["refits"]]


def test_frozen_refit(snapshot_surface):
    # DESIGN section 8 acceptance on every frozen snapshot: after the refit, no violation on the
    # certification grid in either region, the Stage 3 constraints, m in the quoted range, and
    # near-the-money RMSE under 1 vol point on every slice.
    name, surface = snapshot_surface
    assert len(surface.refits) == 12
    params = [r.fit[:5] for r in surface.refits]
    table = arbitrage_table(params, surface.ranges, k_grid(N_CERTIFY))
    counts = ["butterfly quoted", "butterfly extrapolated", "calendar quoted", "calendar extrapolated"]
    print(name)
    print(table.to_string())
    assert (table[counts].fillna(0) == 0).all(axis=None)
    cert = k_grid(N_CERTIFY)
    for expiry, (k, w, weights), T, result, previous in zip(
        surface.expiries, surface.quotes, surface.T, surface.refits, [None, *params[:-1]]
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


def test_frozen_refit_restarts(snapshot_surface):
    # DESIGN section 8 cross-check on every frozen snapshot: the refit repeated from the nine Stage 3 outer
    # starts never beats the chosen refit by more than 1e-10 on f.
    _, surface = snapshot_surface
    previous = None
    for (k, w, weights), result in zip(surface.quotes, surface.refits):
        solved = [r for r in refit_restarts(k, w, weights, previous) if r is not None]
        assert solved
        assert min(r.fit.objective for r in solved) >= result.fit.objective - CROSS_CHECK_TOL
        previous = result.fit[:5]


# Seeds of few-ulp perturbations of the 2026-10-01 market total variances on which, on Windows, the solve from
# the Stage 3 fit alone raised (6, 23 and 29) or converged above the best restart by more than 1e-10 (13 and 28).
PERTURBATION_SEEDS = (6, 13, 23, 28, 29)


@pytest.mark.parametrize("seed", PERTURBATION_SEEDS)
def test_refit_holds_under_few_ulp_perturbations(seed):
    # DESIGN section 8: floating-point results differ between platforms in the last bits, so the 2026-10-01
    # refit is rerun with every market total variance moved by a whole number of ulp drawn from [-4, 4], slice
    # by slice in order of T, and every acceptance check must hold: no violation on the certification grid,
    # the Stage 3 constraints, m in the quoted range and near-the-money RMSE under 1 vol point on every slice.
    chain = pd.read_parquet(FROZEN_CHAIN.with_name("chain_20261001.parquet"))
    kept = chain[chain["status"] == "kept"]
    kept = kept.assign(iv=black76_implied_vol(
        kept["mid"], kept["F"], kept["strike"], kept["T"], kept["D"], kept["option_type"] == "call"
    )).dropna(subset=["iv"])
    rng = np.random.default_rng(seed)
    quotes, fits, T = [], [], []
    for _, rows in kept.groupby("expiry"):
        rows = rows.sort_values("k")
        k, w = rows["k"].to_numpy(), (rows["iv"] ** 2 * rows["T"]).to_numpy()
        w = w * (1.0 + rng.integers(-4, 5, size=w.size) * 2.0**-52)
        weights = vega_weights(rows["F"], rows["strike"], rows["T"], rows["D"], rows["iv"])
        quotes.append((k, w, weights))
        fits.append(fit_svi(k, w, weights))
        T.append(float(rows["T"].iloc[0]))
    refits = refit_surface(quotes, fits)
    assert len(refits) == 12
    params = [r.fit[:5] for r in refits]
    for (k, w, _), T_slice, result, previous in zip(quotes, T, refits, [None, *params[:-1]]):
        _assert_clean(result.fit, k, previous)
        assert svi_fit_errors(k, w, T_slice, result.fit).rmse_near < NEAR_TARGET
    print(f"seed {seed}: restart taken on {sum(r.from_restart for r in refits)} of 12 slices")


# DESIGN section 8 (verified): the test slice's variance-swap vol, 14.9647% by replication and by Gatheral's
# z-integral, against ATM 13.6770%; printed to 4 decimals in percent, so each holds to 5e-7 in vol.
DESIGN_VS, DESIGN_ATM, PRINTED_HALF_UNIT = 0.149647, 0.136770, 5e-7
VIX_CHECK_TOL = 1e-4  # DESIGN section 8: the discrete formula agrees with the replication over its cells within 0.01 vol points


def _design_w(k):
    return raw_svi(k, *DESIGN_SLICE)


def _simpson_truncated(w, k_lo, k_hi, n=2_000_001):
    """2·∫ OTM(K)/K dk over [k_lo, k_hi] by composite Simpson, n points on each side of k = 0.

    Built from black76_price, independently of otm_per_strike and of adaptive quadrature.
    """
    total = 0.0
    for lo, hi, is_call in ((k_lo, 0.0, False), (0.0, k_hi, True)):
        k = np.linspace(lo, hi, n)
        K = np.exp(k)
        price = black76_price(1.0, K, 1.0, 1.0, np.sqrt(w(k)), is_call=is_call)
        total += simpson(2.0 * price / K, x=k)
    return total


def test_bracket():
    T = [0.1, 0.2, 0.5]
    assert bracket(0.15, T) == (0, 1, pytest.approx(0.5, abs=1e-15))
    assert bracket(0.1, T) == (0, 1, 0.0) and bracket(0.5, T) == (1, 2, 1.0)
    assert bracket(0.2, T) == (1, 2, 0.0)  # on a slice, the later pair
    with pytest.raises(ValueError, match="outside"):
        bracket(0.6, T)
    with pytest.raises(ValueError, match="strictly increasing"):
        bracket(0.15, [0.1, 0.1, 0.5])


def test_interpolate_w():
    T_slices = [DESIGN_T, LATER_T, 0.5]
    params = [DESIGN_SLICE, LATER, (0.004, 0.02, -0.4, 0.0, 0.1)]
    k = np.linspace(-0.3, 0.2, 11)
    for T, p in zip(T_slices, params):  # each slice at its own maturity
        np.testing.assert_allclose(interpolate_w(k, T, T_slices, params), raw_svi(k, *p), rtol=1e-15, atol=0)
    # Linear in T at fixed k between the bracketing slices.
    T = DESIGN_T + 0.3 * (LATER_T - DESIGN_T)
    expected = 0.7 * raw_svi(k, *DESIGN_SLICE) + 0.3 * raw_svi(k, *LATER)
    np.testing.assert_allclose(interpolate_w(k, T, T_slices, params), expected, rtol=1e-14, atol=0)
    # Vectorised over a (T, k) grid, and calendar order is kept where the slices are ordered.
    grid_T = np.linspace(DESIGN_T, 0.5, 25)[:, None]
    surface = interpolate_w(k[None, :], grid_T, T_slices, params)
    assert surface.shape == (25, 11)
    np.testing.assert_allclose(surface[7], interpolate_w(k, grid_T[7, 0], T_slices, params), rtol=1e-15, atol=0)
    ordered = (np.diff(np.stack([raw_svi(k, *p) for p in params]), axis=0) >= 0).all(axis=0)
    assert ordered.any() and (np.diff(surface[:, ordered], axis=0) >= 0).all()
    with pytest.raises(ValueError, match="outside"):
        interpolate_w(k, 0.6, T_slices, params)
    with pytest.raises(ValueError, match="one entry per slice"):
        interpolate_w(k, 0.2, T_slices, params[:2])


def test_otm_per_strike():
    k = np.array([-0.4, -0.05, 0.0, 0.05, 0.3])
    w = _design_w(k)
    K = np.exp(k)
    expected = black76_price(1.0, K, 1.0, 1.0, np.sqrt(w), is_call=k >= 0) / K
    np.testing.assert_allclose(otm_per_strike(k, w), expected, rtol=1e-11, atol=1e-16)
    # Deep in the wings it stays finite and non-negative instead of overflowing.
    deep = otm_per_strike(np.array([-60.0, 40.0]), np.array([1.2, 0.8]))
    assert np.isfinite(deep).all() and (deep >= 0).all() and deep.max() < 1e-6
    assert otm_per_strike(0.0, 0.04) == pytest.approx(float(black76_price(1.0, 1.0, 1.0, 1.0, 0.2)), rel=1e-14)


def test_variance_swap_design_values():
    # DESIGN section 8 (verified): 14.9647% by replication and by the z-integral against ATM 13.6770%.
    replication = math.sqrt(variance_swap_replication(_design_w) / DESIGN_T)
    z_integral = math.sqrt(variance_swap_z_integral(_design_w) / DESIGN_T)
    atm = math.sqrt(_design_w(0.0) / DESIGN_T)
    print(f"variance-swap vol {100 * replication:.6f}% by replication, {100 * z_integral:.6f}% by the z-integral, "
          f"ATM {100 * atm:.6f}%")
    for value, printed in ((replication, DESIGN_VS), (z_integral, DESIGN_VS), (atm, DESIGN_ATM)):
        assert abs(value - printed) <= PRINTED_HALF_UNIT
    assert replication == pytest.approx(z_integral, rel=1e-12)


def test_variance_swap_of_a_flat_smile():
    # With a flat smile the variance swap is the implied variance itself.
    flat = lambda k: np.full(np.shape(k), 0.2**2 * 0.5)  # noqa: E731
    assert variance_swap_replication(flat) == pytest.approx(0.02, rel=1e-12)
    assert variance_swap_z_integral(flat) == pytest.approx(0.02, rel=1e-12)


def test_truncated_replication():
    # The truncated integral agrees with an independent high-resolution quadrature on the DESIGN slice,
    # and lies below the full-range value.
    truncated = variance_swap_replication(_design_w, -0.3, 0.1)
    reference = _simpson_truncated(_design_w, -0.3, 0.1)
    print(f"truncated on [-0.3, 0.1]: {truncated:.15e} by quad, {reference:.15e} by Simpson")
    assert abs(truncated - reference) <= 1e-12
    assert truncated < variance_swap_replication(_design_w)
    assert variance_swap_replication(_design_w, -0.3, -0.1) < truncated  # a narrower range, one side only
    with pytest.raises(ValueError, match="k_lo < k_hi"):
        variance_swap_replication(_design_w, 0.1, -0.3)


def test_z_integral_rejects_butterfly_arbitrage():
    # A steep right wing on a level near 0: d_-(k) turns upwards just right of the vertex.
    steep = (1e-4, 1.0, 0.9, 0.3, 0.01)
    with pytest.raises(ValueError, match="d_-"):
        variance_swap_z_integral(lambda k: raw_svi(k, *steep))


def test_vix_discrete_variance():
    # Three nodes by hand: the put at 0.9, the mean of the put and the call at 1 and the call at 1.1.
    w = 0.2**2 * 0.25
    flat = lambda k: np.full(np.shape(k), w)  # noqa: E731
    sigma = math.sqrt(w)
    put, atm, call = (float(black76_price(1.0, x, 1.0, 1.0, sigma, is_call=c))
                      for x, c in ((0.9, False), (1.0, True), (1.1, True)))
    by_hand = 2 * (0.1 / 0.81 * put + 0.1 * atm + 0.1 / 1.21 * call)
    assert vix_discrete_variance(np.log([1.1, 0.9, 1.0, 1.0]), flat) == pytest.approx(by_hand, rel=1e-14)
    # x_0 below the forward: the correction term (1/x_0 - 1)² enters.
    nodes = np.log([0.9, 0.98, 1.06])
    put_098 = float(black76_price(1.0, 0.98, 1.0, 1.0, sigma, is_call=False))
    call_098 = float(black76_price(1.0, 0.98, 1.0, 1.0, sigma, is_call=True))
    call_106 = float(black76_price(1.0, 1.06, 1.0, 1.0, sigma, is_call=True))
    expected = 2 * (0.08 / 0.81 * put + 0.08 / 0.98**2 * 0.5 * (put_098 + call_098)
                    + 0.08 / 1.06**2 * call_106) - (1 / 0.98 - 1) ** 2
    assert vix_discrete_variance(nodes, flat) == pytest.approx(expected, rel=1e-13)
    # On the DESIGN slice it converges to the truncated replication as the strikes get denser.
    truncated = math.sqrt(variance_swap_replication(_design_w, -0.3, 0.1) / DESIGN_T)
    for step, tol in ((1e-3, VIX_CHECK_TOL), (1e-4, 1e-6)):
        nodes = np.linspace(-0.3, 0.1, round(0.4 / step) + 1)
        discrete = math.sqrt(vix_discrete_variance(nodes, _design_w) / DESIGN_T)
        print(f"strike step {step:g} in k: discrete minus replication {discrete - truncated:+.2e} in vol")
        assert abs(discrete - truncated) <= tol
    with pytest.raises(ValueError, match="two distinct nodes"):
        vix_discrete_variance([0.0, 0.0], flat)
    with pytest.raises(ValueError, match="at or below the forward"):
        vix_discrete_variance([0.01, 0.02], flat)


def test_vix_cell_range():
    # Nodes at K/F = 0.9, 1.0, 1.05: the end cells reach half a step out, to 0.85 and 1.075.
    k_lo, k_hi = vix_cell_range(np.log([1.05, 0.9, 1.0, 1.0]))
    assert (k_lo, k_hi) == (pytest.approx(math.log(0.85), rel=1e-14), pytest.approx(math.log(1.075), rel=1e-14))
    # On the DESIGN slice the discrete formula is closer to the replication over its own cells than over
    # the range of its nodes, and that gap shrinks as the strikes get denser.
    quoted = math.sqrt(variance_swap_replication(_design_w, -0.3, 0.1) / DESIGN_T)
    gaps = []
    for step in (1e-3, 1e-4):
        nodes = np.linspace(-0.3, 0.1, round(0.4 / step) + 1)
        discrete = math.sqrt(vix_discrete_variance(nodes, _design_w) / DESIGN_T)
        cells = math.sqrt(variance_swap_replication(_design_w, *vix_cell_range(nodes)) / DESIGN_T)
        print(f"strike step {step:g}: discrete minus replication over its cells {discrete - cells:+.2e}, over "
              f"the range of its nodes {discrete - quoted:+.2e} in vol")
        assert abs(discrete - cells) < abs(discrete - quoted)
        gaps.append(abs(discrete - cells))
    assert gaps[0] <= VIX_CHECK_TOL and gaps[1] <= 1e-6 and gaps[1] < gaps[0] / 10
    with pytest.raises(ValueError, match="two distinct nodes"):
        vix_cell_range([0.0, 0.0])


def test_bridge_30d_matches_the_notebook_path(frozen_surface):
    # On 2026-09-30, bridge_30d computes what the notebook's bridge cell computes, step by step.
    T, quotes = frozen_surface["T"], frozen_surface["quotes"]
    params = [r.fit[:5] for r in frozen_surface["refits"]]
    i, j, lam = bracket(T_BRIDGE, T)
    k_lo, k_hi = max(quotes[i][0].min(), quotes[j][0].min()), min(quotes[i][0].max(), quotes[j][0].max())

    def w30(k):
        return interpolate_w(k, T_BRIDGE, T, params)

    nodes = np.concatenate([quotes[n][0] for n in (i, j)])
    nodes = np.unique(nodes[(nodes >= k_lo) & (nodes <= k_hi)])
    bridge = bridge_30d(surface_from_chain(pd.read_parquet(FROZEN_CHAIN)))
    assert (bridge.i, bridge.j, bridge.lam, bridge.k_lo, bridge.k_hi, bridge.n_strikes) == (i, j, lam, k_lo, k_hi,
                                                                                         nodes.size)
    assert bridge.sigma_vs == math.sqrt(variance_swap_replication(w30, k_lo, k_hi) / T_BRIDGE)
    assert bridge.sigma_vs_discrete == math.sqrt(vix_discrete_variance(nodes, w30) / T_BRIDGE)
    assert bridge.sigma_atm == math.sqrt(float(w30(0.0)) / T_BRIDGE)
    assert bridge.gap == bridge.sigma_vs - bridge.sigma_atm


def test_frozen_bridge(snapshot_surface):
    # DESIGN section 8 on every frozen snapshot: the 30-day slice from the refitted bracketing expiries,
    # over the intersection of their quoted ranges. The CBOE discrete formula at the quoted strikes of both
    # expiries agrees within 0.01 vol points with the replication over the range its cells cover; the
    # truncated value lies below the full-range one, and the full range agrees by replication and by the
    # z-integral.
    name, surface = snapshot_surface
    bridge = bridge_30d(surface)
    assert [f"{surface.expiries[n]:%Y-%m-%d}" for n in (bridge.i, bridge.j)] == ["2026-10-16", "2026-11-20"]
    print(f"{name}: lam = {bridge.lam:.4f}, k in [{bridge.k_lo:.4f}, {bridge.k_hi:.4f}], {bridge.n_strikes} "
          f"strikes: discrete minus replication over its cells {100 * (bridge.sigma_vs_discrete - bridge.sigma_vs_cells):+.5f}, "
          f"over the quoted range {100 * (bridge.sigma_vs_discrete - bridge.sigma_vs):+.4f} vol points")
    assert abs(bridge.sigma_vs_discrete - bridge.sigma_vs_cells) <= VIX_CHECK_TOL
    assert bridge.sigma_vs < bridge.sigma_vs_cells  # the cells reach beyond the quoted range
    assert bridge.sigma_vs < bridge.sigma_vs_full
    assert bridge.sigma_vs_full == pytest.approx(bridge.sigma_vs_full_z, rel=1e-8)
    assert bridge.gap == bridge.sigma_vs - bridge.sigma_atm and bridge.gap > 0
