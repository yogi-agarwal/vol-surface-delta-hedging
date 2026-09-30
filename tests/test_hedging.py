"""Tests for volsurf.hedging: engine mechanics and the DESIGN.md section 4 checks."""

import math

import numpy as np
import pandas as pd
import pytest

from volsurf.black_scholes import bs_delta, bs_gamma, bs_price, bs_vega
from volsurf.data import load_history
from volsurf.hedging import (
    HedgeResult,
    block_bootstrap_indices,
    derman_kamal,
    gbm_paths,
    hedge_grid,
    hedging_error_stats,
    ols_line,
    r2_45,
    r2_ladder,
    simulate_gbm_windows,
    simulate_window,
    simulate_windows,
    stride_subsamples,
    study_windows,
    synthetic_window_set,
    weekday_diagnostic,
    windows_containing,
)

# ---------------------------------------------------------------------------
# Setup (DESIGN.md section 4)
# ---------------------------------------------------------------------------

S0 = 100.0
SIGMA = 0.18
T = 21 / 252
TIMES = np.arange(22) / 252  # 21 daily steps
N_PATHS = 200_000

# Fixed seeds, one per independent sample.
SEED_FLAT = 1  # r = 0 paths for the frictionless and cost checks
SEED_DRIFT = 2  # drift r = 0.04 paths for the carry check
SEED_SET_VOLS = 3  # sigma_i and sigma_true draws for the synthetic window set
SEED_SET_PATHS = 4  # paths for the synthetic window set

INTERVAL_FIELDS = ["interval_pnl", "interval_p_step"]
FIELDS = [f for f in HedgeResult._fields if f not in ["n_intervals", *INTERVAL_FIELDS]]


@pytest.fixture(scope="module")
def flat_paths():
    return gbm_paths(S0, 0.0, SIGMA, TIMES, N_PATHS, seed=SEED_FLAT)


@pytest.fixture(scope="module")
def daily(flat_paths):
    # K = F0 = S0 because r = q = 0.
    return simulate_window(flat_paths, TIMES, 0.0, SIGMA, S0, h=1, c=0.0)


def _derman_kamal(n_intervals):
    """Asymptotic std/C0 of discrete hedging, √(π/4)·vega·σ/(√N·C0), for the ATM call."""
    c0 = bs_price(S0, S0, T, 0.0, 0.0, SIGMA)
    vega = bs_vega(S0, S0, T, 0.0, 0.0, SIGMA)
    return derman_kamal(n_intervals, vega, SIGMA, c0)


# ---------------------------------------------------------------------------
# Path generator
# ---------------------------------------------------------------------------

def test_gbm_paths_shape_and_zero_vol_path():
    t = np.array([0.0, 0.1, 0.25, 0.3])
    paths = gbm_paths(S0, 0.05, np.array([0.0, 0.2, 0.4]), t, 3, seed=0)
    assert paths.shape == (3, 4)
    np.testing.assert_array_equal(paths[:, 0], S0)
    np.testing.assert_allclose(paths[0], S0 * np.exp(0.05 * t), rtol=1e-14)


def test_gbm_paths_reproducible():
    a = gbm_paths(S0, 0.0, SIGMA, TIMES, 10, seed=5)
    b = gbm_paths(S0, 0.0, SIGMA, TIMES, 10, seed=5)
    np.testing.assert_array_equal(a, b)


def test_gbm_paths_log_return_moments():
    # 2.1 million daily log returns: standard errors are about 0.1% of each target.
    mu, dt = 0.04, 1 / 252
    x = np.diff(np.log(gbm_paths(S0, mu, SIGMA, TIMES, 100_000, seed=6)), axis=-1)
    mean_se = SIGMA * math.sqrt(dt / x.size)
    assert x.mean() == pytest.approx((mu - 0.5 * SIGMA**2) * dt, abs=5 * mean_se)
    assert x.var() == pytest.approx(SIGMA**2 * dt, rel=0.01)


# ---------------------------------------------------------------------------
# Hedge grid and R²
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "h, n_intervals", [(1, 21), (2, 11), (3, 7), (5, 5), (7, 3), (10, 3), (21, 1), (30, 1)]
)
def test_hedge_grid(h, n_intervals):
    grid = hedge_grid(21, h)
    assert grid.size - 1 == n_intervals
    assert grid[0] == 0 and grid[-1] == 21
    np.testing.assert_array_equal(np.diff(grid[1:]), h)  # full intervals up to expiry
    assert 0 < grid[1] - grid[0] <= h  # any stub opens the window


def test_hedge_grid_is_anchored_to_expiry():
    np.testing.assert_array_equal(hedge_grid(21, 5), [0, 1, 6, 11, 16, 21])
    np.testing.assert_array_equal(hedge_grid(21, 2)[:3], [0, 1, 3])
    np.testing.assert_array_equal(hedge_grid(21, 7), [0, 7, 14, 21])


def test_hedge_grid_rejects_bad_input():
    with pytest.raises(ValueError):
        hedge_grid(21, 0)
    with pytest.raises(ValueError):
        hedge_grid(0, 1)


def test_r2_45():
    y = np.array([1.0, 2.0, 3.0])
    assert r2_45(y, y) == 1.0
    assert r2_45(y, np.full(3, 2.0)) == 0.0
    assert r2_45(y, np.array([1.0, 2.0, 4.0])) == pytest.approx(0.5, abs=1e-15)
    # A perfect OLS fit with slope 2 still scores below zero about the 45 degree line.
    assert r2_45(y, 2 * y) < 0


# ---------------------------------------------------------------------------
# Engine mechanics against an explicit day-by-day cash account
# ---------------------------------------------------------------------------

def _reference_window(S, t, r, sigma, K, h, c, q, is_call=True):
    """Loop over closes: accrue cash, rebalance on grid days, unwind at the end.

    r holds one rate per step. The premium is carried to expiry in its own
    account, so financing is the interest on the hedge cash alone, and every
    result is divided by that carried premium. Interval P&L comes from marking
    the whole position (option at its Black-Scholes value, stock, cash, less
    the carried premium) just before each rebalance and carrying the mark to
    expiry.
    """
    n = len(S) - 1
    expiry = t[-1]

    def to_expiry(a):
        """Growth of one unit of cash from close a to close n, day by day."""
        return math.prod(math.exp(r[j] * (t[j + 1] - t[j])) for j in range(a, n))

    c0 = bs_price(S[0], K, expiry - t[0], r[0], q, sigma, is_call)
    premium, cash, position = c0, 0.0, 0.0
    stock = financing = costs = turnover = 0.0
    set_days, marks = [], []
    payoff = max(S[n] - K, 0.0) if is_call else max(K - S[n], 0.0)
    for j in range(n + 1):
        if j > 0:
            step_growth = math.exp(r[j - 1] * (t[j] - t[j - 1]))
            premium *= step_growth
            interest = cash * (step_growth - 1.0)
            financing += interest
            cash += interest
            stock += position * (S[j] - S[j - 1])
        if j == n:
            target = 0.0
        elif j == 0 or (n - j) % h == 0:
            target = -bs_delta(S[j], K, expiry - t[j], r[j], q, sigma, is_call)
            set_days.append(j)
            option = bs_price(S[j], K, expiry - t[j], r[j], q, sigma, is_call)
            marks.append(to_expiry(j) * (option + position * S[j] + cash - premium))
        else:
            target = position
        trade = target - position
        if 0 < j < n:
            turnover += abs(trade)
        cost = c * abs(trade) * S[j]
        cash -= trade * S[j] + cost
        costs -= cost
        position = target
    pnl = cash + payoff - premium
    interval_pnl = np.diff(marks + [pnl])

    rv = sum(math.log(S[j + 1] / S[j]) ** 2 for j in range(n))
    var_gap = rv / (expiry - t[0]) - sigma**2
    ends = set_days[1:] + [n]
    p_path = p_step = 0.0
    interval_p_step = []
    for a, b in zip(set_days, ends):
        dollar_gamma = to_expiry(a) * bs_gamma(S[a], K, expiry - t[a], r[a], q, sigma) * S[a] ** 2
        dt = t[b] - t[a]
        p_path += 0.5 * var_gap * dollar_gamma * dt
        interval_p_step.append(0.5 * dollar_gamma * ((S[b] / S[a] - 1.0) ** 2 - sigma**2 * dt))
        p_step += interval_p_step[-1]
    vega0 = bs_vega(S[0], K, expiry - t[0], r[0], q, sigma)
    p_gap = to_expiry(0) * vega0 * var_gap / (2 * sigma)
    return {
        "interval_pnl": interval_pnl / premium,
        "interval_p_step": np.array(interval_p_step) / premium,
        "pnl": pnl / premium,
        "payoff": payoff / premium,
        "stock": stock / premium,
        "financing": financing / premium,
        "costs": costs / premium,
        "turnover": turnover,
        "p_gap": p_gap / premium,
        "p_path": p_path / premium,
        "p_step": p_step / premium,
        "rv": rv,
        "c0": c0,
    }


@pytest.fixture(scope="module")
def uneven_window():
    """Five paths on an irregular calendar grid with daily step rates and per-path inputs."""
    days = np.array([0, 1, 2, 3, 6, 7, 8, 9, 10, 13, 14])  # weekends skipped
    t = days / 365
    rng = np.random.default_rng(7)
    r = 0.03 + 0.01 * rng.standard_normal(t.size)
    sigma = np.array([0.12, 0.18, 0.25, 0.4, 0.18])
    S = gbm_paths(400.0, 0.03, sigma * 1.3, t, 5, seed=8)
    K = np.array([380.0, 400.0, 405.0, 420.0, 401.0])
    q = np.array([0.0, 0.013, 0.0, 0.02, 0.0])
    return S, t, r[:-1], sigma, K, q  # one rate per step; the rate at the last close is never used


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("h", [1, 3, 4, 10, 12])
def test_matches_reference_loop(uneven_window, h, is_call):
    S, t, r, sigma, K, q = uneven_window
    c = 0.0025
    res = simulate_window(S, t, r, sigma, K, h=h, c=c, q=q, is_call=is_call)
    assert res.n_intervals == hedge_grid(len(t) - 1, h).size - 1
    for i in range(S.shape[0]):
        ref = _reference_window(S[i], t, r, sigma[i], K[i], h, c, q[i], is_call)
        for name in INTERVAL_FIELDS:
            # The reference differences marks of the whole position (stock and cash included), so
            # its rounding grows with the position relative to the premium (intervals reach 30
            # premiums on the cheap out-of-the-money put): relative 1e-12 as for the other
            # fields, with a floor of 1e-12 of the premium for intervals near zero.
            value = ref.pop(name)
            assert getattr(res, name)[i].shape == (res.n_intervals,)
            np.testing.assert_allclose(getattr(res, name)[i], value, rtol=1e-12, atol=1e-12, err_msg=name)
        for name, value in ref.items():
            assert getattr(res, name)[i] == pytest.approx(value, rel=1e-12, abs=1e-14), name


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("h", [1, 3, 10])
def test_intervals_sum_to_window(uneven_window, h, is_call):
    S, t, r, sigma, K, q = uneven_window
    res = simulate_window(S, t, r, sigma, K, h=h, c=0.001, q=q, is_call=is_call)
    assert res.interval_pnl.shape == res.interval_p_step.shape == (5, res.n_intervals)
    np.testing.assert_allclose(res.interval_pnl.sum(axis=-1), res.pnl, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.interval_p_step.sum(axis=-1), res.p_step, rtol=0, atol=1e-12)


def test_components_sum_to_pnl(uneven_window):
    S, t, r, sigma, K, q = uneven_window
    res = simulate_window(S, t, r, sigma, K, h=2, c=0.001, q=q)
    total = res.payoff + res.premium + res.stock + res.financing + res.costs
    np.testing.assert_allclose(total, res.pnl, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(res.premium, -1.0)


def test_broadcasting_matches_single_path(uneven_window):
    S, t, r, sigma, K, q = uneven_window
    res = simulate_window(S, t, r, sigma, K, h=2, c=0.0005, q=q)
    one = simulate_window(S[3], t, r, sigma[3], K[3], h=2, c=0.0005, q=q[3])
    for name in ("pnl", "stock", "financing", "costs", "turnover", "p_gap", "p_path", "p_step"):
        assert np.ndim(getattr(one, name)) == 0
        assert getattr(res, name)[3] == pytest.approx(getattr(one, name), rel=1e-14, abs=1e-15)
    for name in INTERVAL_FIELDS:
        assert getattr(one, name).shape == (one.n_intervals,)
        np.testing.assert_allclose(getattr(res, name)[3], getattr(one, name), rtol=1e-14, atol=1e-15)


def test_static_hedge_has_no_interior_turnover(uneven_window):
    S, t, r, sigma, K, q = uneven_window
    res = simulate_window(S, t, r, sigma, K, h=len(t) - 1, c=0.001, q=q)
    assert res.n_intervals == 1
    np.testing.assert_array_equal(res.turnover, 0.0)


def test_rejects_non_increasing_times():
    with pytest.raises(ValueError):
        simulate_window([100.0, 101.0, 99.0], [0.0, 0.1, 0.1], 0.0, 0.2, 100.0, h=1, c=0.0)


# ---------------------------------------------------------------------------
# Input shapes and windows of different lengths
# ---------------------------------------------------------------------------

def test_per_window_times_and_rates(uneven_window):
    # Each path gets its own calendar and rates, as real windows starting on different days do.
    S, t, r, sigma, K, q = uneven_window
    times = t * (1.0 + 0.1 * np.arange(5))[:, None] + 0.5
    rates = r + 0.002 * np.arange(5)[:, None]
    assert times.shape == (5, 11) and rates.shape == (5, 10)
    res = simulate_window(S, times, rates, sigma, K, h=3, c=0.0005, q=q)
    for i in range(5):
        one = simulate_window(S[i], times[i], rates[i], sigma[i], K[i], h=3, c=0.0005, q=q[i])
        for name in FIELDS + INTERVAL_FIELDS:
            assert getattr(res, name)[i] == pytest.approx(getattr(one, name), rel=1e-14, abs=1e-15), name


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda S, t, r: {"r": np.append(r, r[-1])}, id="r per close"),
        pytest.param(lambda S, t, r: {"r": np.full(5, 0.03)}, id="r per path"),
        pytest.param(lambda S, t, r: {"r": np.tile(r, (4, 1))}, id="r wrong path count"),
        pytest.param(lambda S, t, r: {"r": r[None, None, :]}, id="r 3-D"),
        pytest.param(lambda S, t, r: {"t": t[:-1]}, id="t wrong length"),
        pytest.param(lambda S, t, r: {"t": np.tile(t, (4, 1))}, id="t wrong path count"),
        pytest.param(lambda S, t, r: {"sigma_i": np.full(11, 0.2)}, id="sigma_i per close"),
        pytest.param(lambda S, t, r: {"sigma_i": np.full((5, 1), 0.2)}, id="sigma_i 2-D"),
        pytest.param(lambda S, t, r: {"K": np.full(6, 400.0)}, id="K wrong path count"),
        pytest.param(lambda S, t, r: {"q": np.zeros((5, 10))}, id="q 2-D"),
        pytest.param(lambda S, t, r: {"S": S[None]}, id="S 3-D"),
        pytest.param(lambda S, t, r: {"S": S[:, :1], "t": t[:1], "r": 0.03}, id="single close"),
        pytest.param(
            lambda S, t, r: {"S": S[0], "sigma_i": np.full(5, 0.2), "K": 400.0, "q": 0.0},
            id="per-path sigma_i with one path",
        ),
    ],
)
def test_rejects_ambiguous_shapes(uneven_window, bad):
    S, t, r, sigma, K, q = uneven_window
    args = {"S": S, "t": t, "r": r, "sigma_i": sigma, "K": K, "q": q}
    args.update(bad(S, t, r))
    with pytest.raises(ValueError):
        simulate_window(h=2, c=0.0, **args)


def test_simulate_windows_groups_by_length(uneven_window):
    S, t, r, sigma, K, _ = uneven_window
    history, rates = S[0], np.append(r, r[-1])  # one rate per close; no window reads the last
    starts = np.array([0, 1, 2, 3, 5])
    ends = np.array([4, 6, 6, 10, 9])  # lengths 4, 5, 4, 7, 4
    q = 0.013
    res = simulate_windows(history, t, rates, starts, ends, sigma, K, h=2, c=0.001, q=q)
    assert res.pnl.shape == (5,)
    n_max = 4  # ceil(7/2) for the longest window
    assert res.interval_pnl.shape == res.interval_p_step.shape == (5, n_max)
    for w, (a, b) in enumerate(zip(starts, ends)):
        one = simulate_window(
            history[a : b + 1], t[a : b + 1], rates[a:b], sigma[w], K[w], h=2, c=0.001, q=q
        )
        assert res.n_intervals[w] == one.n_intervals
        for name in FIELDS:
            assert getattr(res, name)[w] == pytest.approx(getattr(one, name), rel=1e-14, abs=1e-15), name
        # Interval columns are aligned at expiry and NaN before the window's first interval.
        pad = n_max - one.n_intervals
        for name in INTERVAL_FIELDS:
            assert np.isnan(getattr(res, name)[w, :pad]).all()
            np.testing.assert_allclose(getattr(res, name)[w, pad:], getattr(one, name), rtol=1e-14, atol=1e-15)
        # Column j ends at close ends[w] - (n_max - 1 - j)·h, as documented.
        documented_ends = b - (n_max - 1 - np.arange(pad, n_max)) * 2
        np.testing.assert_array_equal(documented_ends, a + hedge_grid(b - a, 2)[1:])


@pytest.mark.parametrize(
    "starts, ends",
    [([0, 3], [4, 3]), ([0], [11]), ([-1], [4]), ([0.0], [4.0]), ([0, 1], [4])],
    ids=["empty window", "past the end", "negative start", "float bounds", "length mismatch"],
)
def test_simulate_windows_rejects_bad_bounds(uneven_window, starts, ends):
    S, t, r, *_ = uneven_window
    with pytest.raises(ValueError):
        simulate_windows(S[0], t, np.append(r, r[-1]), starts, ends, 0.2, 400.0, h=1, c=0.0)


# ---------------------------------------------------------------------------
# Money units and puts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [1, 3])
def test_money_units_invariant_to_deterministic_rate(uneven_window, h):
    # With a flat rate r, the window S' = S·exp(r(t - t0)), K' = K·exp(r·T0) is the r = 0
    # window in forward terms: deltas, C0 and vega0 are unchanged, and every cash flow and
    # dollar-gamma term carried to expiry scales by exp(r·T0). P&L over C0 grown to expiry,
    # and predictors weighted by growth to expiry, must therefore agree. Dividing by C0
    # alone, or leaving the predictor terms unweighted, misses by about 0.4% at r = 0.05.
    S, t, _, sigma, K, q = uneven_window
    r, c = 0.05, 5e-4
    T0 = t[-1] - t[0]
    base = simulate_window(S, t, 0.0, sigma, K, h=h, c=c, q=q)
    fwd = simulate_window(S * np.exp(r * (t - t[0])), t, r, sigma, K * math.exp(r * T0), h=h, c=c, q=q)
    pnl_gap = np.max(np.abs(fwd.pnl - base.pnl))
    ratio_gap = np.max(np.abs(fwd.p_path / fwd.p_gap / (base.p_path / base.p_gap) - 1.0))
    print(f"h = {h}: max |pnl' - pnl| = {pnl_gap:.1e}, max P_path/P_gap relative gap = {ratio_gap:.1e}")
    np.testing.assert_allclose(fwd.c0, base.c0, rtol=1e-12)
    np.testing.assert_allclose(fwd.pnl, base.pnl, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fwd.p_path / fwd.p_gap, base.p_path / base.p_gap, rtol=1e-10)


@pytest.mark.parametrize("h", [1, 3, 10])
def test_hedged_call_put_parity(uneven_window, h):
    # Long call minus long put is a forward whose delta is exactly one share when q = 0, so the
    # two hedged positions differ by a hedged forward, which earns nothing at a flat rate. Their
    # P&L carried to expiry, pnl·C0·G0 with G0 common to both, must agree path by path. Costs are
    # off because the opening and closing trades differ in size.
    S, t, _, sigma, K, _ = uneven_window
    call = simulate_window(S, t, 0.04, sigma, K, h=h, c=0.0, q=0.0, is_call=True)
    put = simulate_window(S, t, 0.04, sigma, K, h=h, c=0.0, q=0.0, is_call=False)
    gap = (call.pnl * call.c0 - put.pnl * put.c0) / call.c0
    print(f"h = {h}: max |call - put| hedged P&L = {np.max(np.abs(gap)):.1e} of C0·G0")
    np.testing.assert_allclose(gap, 0.0, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Discrete hedging checks (DESIGN.md section 4)
# ---------------------------------------------------------------------------

def test_daily_mean_pnl(daily):
    mean = daily.pnl.mean()
    print(f"daily mean P&L/C0 = {mean:.5f}")
    assert abs(mean) <= 0.005


def test_daily_std_pnl(daily):
    std = daily.pnl.std()
    dk = _derman_kamal(21)
    print(f"daily std P&L/C0 = {std:.5f}, Derman-Kamal = {dk:.6f}")
    assert dk == pytest.approx(0.193347, abs=5e-7)
    assert 0.175 <= std <= 0.200


def test_every_2_days_std_pnl(flat_paths):
    # DESIGN.md gives "about 0.2566" on the expiry-anchored grid (one-day stub
    # first); tolerance 0.005 is several Monte Carlo standard errors yet
    # excludes the Derman-Kamal value 0.273.
    res = simulate_window(flat_paths, TIMES, 0.0, SIGMA, S0, h=2, c=0.0)
    std = res.pnl.std()
    dk = _derman_kamal(21 / 2)  # asymptotic formula with interval 2/252
    print(f"every-2-days std P&L/C0 = {std:.5f}, Derman-Kamal = {dk:.6f}")
    assert dk == pytest.approx(0.273, abs=5e-4)
    assert std == pytest.approx(0.2566, abs=0.005)


def test_daily_interior_turnover(daily):
    target = math.sqrt(21) / math.pi
    turnover = daily.turnover.mean()
    print(f"mean interior turnover = {turnover:.5f}, sqrt(N)/pi = {target:.6f}")
    assert target == pytest.approx(1.458679, abs=5e-7)
    assert turnover == pytest.approx(target, rel=0.05)


def test_daily_5bp_costs(flat_paths):
    # Std tolerance as for the every-2-days check.
    res = simulate_window(flat_paths, TIMES, 0.0, SIGMA, S0, h=1, c=5e-4)
    mean, std = res.pnl.mean(), res.pnl.std()
    print(f"5 bp daily: mean P&L/C0 = {mean:.5f}, std = {std:.5f}")
    assert mean == pytest.approx(-0.060, abs=0.005)
    assert std == pytest.approx(0.184, abs=0.005)


@pytest.mark.parametrize("q, expected, exact_value", [(0.013, 0.0269, 0.026907), (0.0, 0.0, 0.0)])
def test_carry_consistency(q, expected, exact_value):
    # Total-return path with drift r; K is the pricing model's forward S0·exp((r - q)T).
    # The hedge has zero mean under Q whatever delta is used, so the mean P&L over C0
    # grown to expiry is exactly C(q = 0)/C(q) - 1.
    r = 0.04
    paths = gbm_paths(S0, r, SIGMA, TIMES, N_PATHS, seed=SEED_DRIFT)
    K = S0 * math.exp((r - q) * T)
    exact = bs_price(S0, K, T, r, 0.0, SIGMA) / bs_price(S0, K, T, r, q, SIGMA) - 1.0
    res = simulate_window(paths, TIMES, r, SIGMA, K, h=1, c=0.0, q=q)
    mean = res.pnl.mean()
    print(f"carry q = {q}: mean P&L/(C0·G0) = {mean:.6f}, exact = {exact:.6f}")
    assert exact == pytest.approx(exact_value, abs=5e-7)
    assert mean == pytest.approx(expected, abs=0.003)


def test_p_step_r2_on_synthetic_window_set():
    # DESIGN.md section 3: 3,000 windows, sigma_i ~ lognormal(ln 0.17, 0.35),
    # sigma_true = sigma_i·exp(N(-0.2, 0.3)), daily hedging, r = q = 0, K = F0.
    paths, times, sigma_i = synthetic_window_set(3000, SEED_SET_VOLS, SEED_SET_PATHS)
    np.testing.assert_array_equal(times, TIMES)
    res = simulate_window(paths, times, 0.0, sigma_i, S0, h=1, c=0.0)
    ladder = [r2_45(res.pnl, p) for p in (res.p_gap, res.p_path, res.p_step)]
    print("daily R2_45 P_gap / P_path / P_step = " + " / ".join(f"{x:.4f}" for x in ladder))
    assert ladder[2] >= 0.95


def test_synthetic_window_set():
    S, t, sigma_i = synthetic_window_set(50, 3, 4)
    assert S.shape == (50, 22) and t.shape == (22,) and sigma_i.shape == (50,)
    np.testing.assert_array_equal(S[:, 0], 100.0)
    rng = np.random.default_rng(3)
    np.testing.assert_allclose(sigma_i, np.exp(rng.normal(math.log(0.17), 0.35, 50)), rtol=1e-15)
    np.testing.assert_array_equal(S, synthetic_window_set(50, 3, 4)[0])


# ---------------------------------------------------------------------------
# GBM paths on real calendars
# ---------------------------------------------------------------------------

def test_gbm_paths_per_path_grids_and_drift():
    t = np.array([[0, 1, 2, 5, 6], [0, 3, 4, 5, 6]]) / 365
    mu = np.array([[0.01, 0.02, 0.03, 0.04], [0.05, 0.0, -0.01, 0.02]])
    paths = gbm_paths(S0, mu, np.zeros(2), t, 2, seed=0)
    growth = np.cumsum(mu * np.diff(t, axis=-1), axis=-1)
    np.testing.assert_allclose(paths[:, 1:], S0 * np.exp(growth), rtol=1e-14)
    # Identical rows of a 2-D grid, with a per-step drift, reproduce the 1-D grid.
    one = gbm_paths(S0, 0.03, SIGMA, TIMES, 4, seed=[9, 1])
    two = gbm_paths(S0, np.full((4, 21), 0.03), SIGMA, np.tile(TIMES, (4, 1)), 4, seed=[9, 1])
    np.testing.assert_allclose(two, one, rtol=1e-14)


@pytest.fixture(scope="module")
def calendar_history():
    """Business days from late September 2021 to mid January 2022, less two holidays."""
    dates = pd.bdate_range("2021-09-27", "2022-01-14")
    dates = dates[~dates.isin(pd.to_datetime(["2021-11-25", "2021-12-24"]))]
    rng = np.random.default_rng(10)
    return pd.DataFrame(
        {
            "spy": 400.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(dates.size))),
            "sigma_i": 0.15 + 0.05 * rng.random(dates.size),
            "r": 0.03 + 0.01 * rng.random(dates.size),
        },
        index=dates,
    )


def test_simulate_gbm_windows_matches_engine(calendar_history):
    # One window of each length, so each length group holds one window and its paths can be
    # redrawn here with the documented seed [seed, n].
    win = study_windows(calendar_history)
    r = calendar_history["r"].to_numpy()
    lengths = win.ends - win.starts
    sel = np.array([np.flatnonzero(lengths == n)[0] for n in np.unique(lengths)])
    assert sel.size >= 2
    starts, ends, sigma_i = win.starts[sel], win.ends[sel], win.sigma_i[sel]
    sigma_true = np.linspace(0.1, 0.3, sel.size)
    n_paths, h, c = 3, 2, 5e-4
    res = simulate_gbm_windows(win.t, r, starts, ends, sigma_true, sigma_i, h, c, n_paths, seed=15)
    n_max = res.interval_pnl.shape[-1]
    assert res.pnl.shape == (sel.size, n_paths) and res.interval_pnl.shape == (sel.size, n_paths, n_max)
    for w, (a, b) in enumerate(zip(starts, ends)):
        times, rates = np.tile(win.t[a : b + 1], (n_paths, 1)), np.tile(r[a:b], (n_paths, 1))
        paths = gbm_paths(100.0, rates, np.full(n_paths, sigma_true[w]), times, n_paths, seed=[15, b - a])
        strike = 100.0 * math.exp(r[a] * (win.t[b] - win.t[a]))
        one = simulate_window(paths, times, rates, sigma_i[w], strike, h, c)
        assert res.n_intervals[w] == one.n_intervals
        for name in FIELDS:
            np.testing.assert_allclose(getattr(res, name)[w], getattr(one, name), rtol=1e-14, atol=1e-15, err_msg=name)
        pad = n_max - one.n_intervals
        for name in INTERVAL_FIELDS:
            assert np.isnan(getattr(res, name)[w, :, :pad]).all()
            np.testing.assert_allclose(getattr(res, name)[w, :, pad:], getattr(one, name), rtol=1e-14, atol=1e-15)


# ---------------------------------------------------------------------------
# Windows and subsamples of the real-data study
# ---------------------------------------------------------------------------

def _check_window_rules(history, win, first_start):
    """Assert the DESIGN.md section 3 window rules for windows built from history."""
    dates = history.index
    assert dates[win.starts[0]] == dates[dates >= first_start][0]
    np.testing.assert_array_equal(np.diff(win.starts), 1)  # one window per trading day
    horizon = dates[win.starts] + pd.Timedelta(days=30)
    assert (dates[win.ends] <= horizon).all()
    inside = win.ends + 1 < dates.size
    assert (dates[win.ends[inside] + 1] > horizon[inside]).all()  # t_end is the last day on or before
    assert horizon[-1] <= dates[-1] < dates[win.starts[-1] + 1] + pd.Timedelta(days=30)
    np.testing.assert_allclose(win.T0, (dates[win.ends] - dates[win.starts]).days / 365, rtol=1e-12)
    S, r = history["spy"].to_numpy(), history["r"].to_numpy()
    np.testing.assert_allclose(win.K, S[win.starts] * np.exp(r[win.starts] * win.T0), rtol=1e-15)
    np.testing.assert_array_equal(win.sigma_i, history["sigma_i"].to_numpy()[win.starts])


def test_study_windows_on_calendar(calendar_history):
    win = study_windows(calendar_history, first_start="2021-10-02")  # a Saturday
    dates = calendar_history.index
    assert dates[win.starts[0]] == pd.Timestamp("2021-10-04")
    _check_window_rules(calendar_history, win, "2021-10-02")
    np.testing.assert_allclose(win.t, (dates - dates[0]).days / 365, rtol=0, atol=1e-15)
    # 2021-10-26 plus 30 days is Thanksgiving, a holiday here, so t_end is the day before.
    w = np.flatnonzero(dates[win.starts] == pd.Timestamp("2021-10-26"))[0]
    assert dates[win.ends[w]] == pd.Timestamp("2021-11-24")


def test_study_windows_on_frozen_history():
    history = load_history()
    win = study_windows(history)
    assert history.index[win.starts[0]] == pd.Timestamp("2021-10-01")
    _check_window_rules(history, win, "2021-10-01")
    steps = win.ends - win.starts
    print(f"{win.starts.size} windows, {steps.min()} to {steps.max()} steps")
    assert steps.min() >= 17 and steps.max() <= 22


def test_windows_containing(calendar_history):
    dates = calendar_history.index
    starts = np.array([0, 5, 10, 20])
    ends = starts + 4
    np.testing.assert_array_equal(
        windows_containing(dates, starts, ends, dates[8], dates[11]), [False, True, True, False]
    )
    # A window ending on the first day, or starting on the last, is flagged.
    np.testing.assert_array_equal(
        windows_containing(dates, [4, 11], [8, 15], dates[8], dates[11]), [True, True]
    )
    # A range holding no trading day (a weekend) flags nothing.
    assert not windows_containing(dates, starts, ends, "2021-10-02", "2021-10-03").any()


def _check_stride_subsamples(win, stride):
    """Assert that stride subsamples partition the windows and share no daily return."""
    subs = stride_subsamples(win.starts, win.ends, stride)
    assert len(subs) == stride
    np.testing.assert_array_equal(np.sort(np.concatenate(subs)), np.arange(win.starts.size))
    for k, sub in enumerate(subs):
        assert sub[0] == k
        np.testing.assert_array_equal(np.diff(sub), stride)
        assert (win.starts[sub[1:]] >= win.ends[sub[:-1]]).all()  # no shared daily return
    return subs


def test_stride_subsamples(calendar_history):
    win = study_windows(calendar_history)
    longest = int((win.ends - win.starts).max())
    _check_stride_subsamples(win, longest)  # the smallest stride with no overlap
    _check_stride_subsamples(win, longest + 3)
    # One step shorter than the longest window, the longest one overlaps the window after it.
    with pytest.raises(ValueError):
        stride_subsamples(win.starts, win.ends, longest - 1)
    with pytest.raises(ValueError):
        stride_subsamples(win.starts, win.ends, 0)


def test_stride_22_subsamples_on_frozen_history():
    win = study_windows(load_history())
    subs = _check_stride_subsamples(win, 22)
    print(f"22 subsamples of {min(s.size for s in subs)} to {max(s.size for s in subs)} windows")


# ---------------------------------------------------------------------------
# R² ladder, bootstrap and hedging error
# ---------------------------------------------------------------------------

def test_r2_45_along_last_axis():
    y = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    p = np.array([[1.0, 2.0, 4.0], [2.0, 2.0, 2.0]])
    np.testing.assert_allclose(r2_45(y, p), [0.5, 0.0], rtol=0, atol=1e-15)


def test_ols_line():
    p = np.linspace(-1.0, 2.0, 7)
    alpha, beta = ols_line(0.3 + 1.7 * p, p)
    assert alpha == pytest.approx(0.3, abs=1e-14) and beta == pytest.approx(1.7, rel=1e-14)
    y = 0.5 * p + np.random.default_rng(11).standard_normal(7)
    np.testing.assert_allclose(ols_line(y, p), np.polyfit(p, y, 1)[::-1], rtol=1e-12)
    _, betas = ols_line(np.stack([y, 0.3 + 1.7 * p]), np.stack([p, p]))  # rows are samples
    np.testing.assert_allclose(betas, [ols_line(y, p)[1], 1.7], rtol=1e-12)


def test_block_bootstrap_indices():
    idx = block_bootstrap_indices(100, 42, 500, seed=12)
    assert idx.shape == (500, 100)
    np.testing.assert_array_equal(idx, block_bootstrap_indices(100, 42, 500, seed=12))
    # Each row joins ceil(100/42) = 3 runs of consecutive indices, cut at 42 and 84.
    for start in (0, 42, 84):
        np.testing.assert_array_equal(np.diff(idx[:, start : start + 42], axis=1), 1)
    # Block starts are drawn from every position 0, ..., 100 - 42 and no other.
    firsts = idx[:, [0, 42, 84]]
    assert firsts.min() == 0 and firsts.max() == 58
    assert idx.min() >= 0 and idx.max() <= 99
    with pytest.raises(ValueError):
        block_bootstrap_indices(10, 11, 5, seed=0)


def test_r2_ladder():
    rng = np.random.default_rng(13)
    p = rng.standard_normal(200)
    y = p + 0.5 * rng.standard_normal(200)
    boot = block_bootstrap_indices(200, 10, 300, seed=14)
    table = r2_ladder(y, {"perfect": y, "noisy": p}, boot)
    assert list(table.index) == ["perfect", "noisy"]
    perfect, noisy = table.loc["perfect"], table.loc["noisy"]
    assert perfect["r2_45"] == 1.0 and perfect["r2_lo"] == 1.0 and perfect["r2_hi"] == 1.0
    assert perfect["alpha"] == pytest.approx(0.0, abs=1e-15)
    for name in ("beta", "beta_lo", "beta_hi"):
        assert perfect[name] == pytest.approx(1.0, rel=1e-14)
    assert noisy["n"] == 200
    assert noisy["r2_45"] == pytest.approx(r2_45(y, p), rel=1e-14)
    assert (noisy["alpha"], noisy["beta"]) == pytest.approx(ols_line(y, p), rel=1e-14)
    assert noisy["r2_lo"] < noisy["r2_45"] < noisy["r2_hi"]
    assert noisy["beta_lo"] < noisy["beta"] < noisy["beta_hi"]
    assert r2_ladder(y, {"noisy": p})[["r2_lo", "r2_hi", "beta_lo", "beta_hi"]].isna().all(axis=None)


def test_hedging_error_stats():
    pnl = np.array([[0.1, -0.2], [0.3, 0.05]])
    p_gap = np.array([[0.0, -0.1], [0.1, 0.1]])
    e = (pnl - p_gap).ravel()
    stats = hedging_error_stats(pnl, p_gap)
    assert stats["mean"] == pytest.approx(0.0375, abs=1e-15)
    assert stats["std"] == pytest.approx(e.std(ddof=0), rel=1e-14)
    assert stats["rmse"] ** 2 == pytest.approx(np.mean(e**2), rel=1e-14)


# ---------------------------------------------------------------------------
# Weekday diagnostic
# ---------------------------------------------------------------------------

def test_weekday_diagnostic(calendar_history):
    dates = calendar_history.index
    win = study_windows(calendar_history)
    S, r = calendar_history["spy"].to_numpy(), calendar_history["r"].to_numpy()
    res = simulate_windows(S, win.t, r, win.starts, win.ends, win.sigma_i, win.K, h=1, c=0.0)

    # Constructed residuals: 1 on intervals closing on a Monday, 0 on the rest.
    valid = ~np.isnan(res.interval_pnl)
    end_close = win.ends[:, None] - np.arange(valid.shape[1])[::-1]
    monday = valid & (dates.weekday.to_numpy()[np.where(valid, end_close, 0)] == 0)
    pnl = np.where(valid, res.interval_p_step + monday, np.nan)
    same = np.tile(np.arange(win.starts.size), (4, 1))  # resamples equal to the sample
    table, eta2 = weekday_diagnostic(pnl, res.interval_p_step, dates, win.starts, win.ends, 1, same)

    assert list(table.index) == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "All"]
    n_all = valid.sum()
    assert table.loc["All", "intervals"] == n_all == table["intervals"].iloc[:5].sum()
    np.testing.assert_allclose(table["residual"], [1, 0, 0, 0, 0, monday.sum() / n_all], atol=1e-12)
    np.testing.assert_allclose(table["share"], [1, 0, 0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(table["residual_lo"], table["residual"], atol=1e-12)
    np.testing.assert_allclose(table["residual_hi"], table["residual"], atol=1e-12)
    assert eta2 == pytest.approx(1.0, abs=1e-12)
    assert table.loc["All", "pnl"] == pytest.approx(np.nanmean(pnl), rel=1e-12)
    assert table.loc["All", "p_step"] == pytest.approx(np.nanmean(res.interval_p_step), rel=1e-12)
    # Daily intervals span one calendar day, three over a weekend, more over a holiday.
    assert table.loc["Tuesday", "calendar_days"] == 1.0 and table.loc["Wednesday", "calendar_days"] == 1.0
    assert table.loc["Monday", "calendar_days"] > 3.0  # 2021-12-27 follows the 12-24 holiday

    # With a stub (h = 2), interval lengths still add up to each window's calendar days.
    res2 = simulate_windows(S, win.t, r, win.starts, win.ends, win.sigma_i, win.K, h=2, c=0.0)
    table2, _ = weekday_diagnostic(
        res2.interval_pnl, res2.interval_p_step, dates, win.starts, win.ends, 2
    )
    total_days = np.sum((dates[win.ends] - dates[win.starts]).days)
    assert table2.loc["All", "calendar_days"] == pytest.approx(total_days / res2.n_intervals.sum(), rel=1e-12)
    assert table2["residual_lo"].isna().all()
