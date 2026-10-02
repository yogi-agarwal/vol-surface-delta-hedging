import math
from decimal import Decimal

import numpy as np
import pytest
from scipy.integrate import quad

from volsurf.arbitrage import (
    N_CERTIFY,
    N_FINE,
    N_REFIT,
    arbitrage_table,
    butterfly_g,
    k_grid,
    svi_density,
    svi_derivatives,
    violation_counts,
)
from volsurf.black_scholes import black76_price
from volsurf.svi import raw_svi

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
