"""Tests for volsurf.hedging: engine mechanics and the DESIGN.md section 4 checks."""

import math

import numpy as np
import pytest

from volsurf.black_scholes import bs_delta, bs_gamma, bs_price, bs_vega
from volsurf.hedging import gbm_paths, hedge_grid, r2_45, simulate_window

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
    return math.sqrt(math.pi / 4) * vega * SIGMA / (math.sqrt(n_intervals) * c0)


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
    np.testing.assert_array_equal(np.diff(grid[:-1]), h)
    assert 0 < grid[-1] - grid[-2] <= h


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

def _reference_window(S, t, r, sigma, K, h, c, q):
    """Loop over closes: accrue cash, rebalance on grid days, unwind at the end."""
    n = len(S) - 1
    expiry = t[-1]
    c0 = bs_price(S[0], K, expiry - t[0], r[0], q, sigma)
    cash, position = -c0, 0.0
    stock = financing = costs = turnover = 0.0
    set_days = []
    for j in range(n + 1):
        if j > 0:
            interest = cash * (math.exp(r[j - 1] * (t[j] - t[j - 1])) - 1.0)
            financing += interest
            cash += interest
            stock += position * (S[j] - S[j - 1])
        if j == n:
            target = 0.0
        elif j % h == 0:
            target = -bs_delta(S[j], K, expiry - t[j], r[j], q, sigma)
            set_days.append(j)
        else:
            target = position
        trade = target - position
        if 0 < j < n:
            turnover += abs(trade)
        cost = c * abs(trade) * S[j]
        cash -= trade * S[j] + cost
        costs -= cost
        position = target
    payoff = max(S[n] - K, 0.0)
    pnl = cash + payoff

    rv = sum(math.log(S[j + 1] / S[j]) ** 2 for j in range(n))
    var_gap = rv / (expiry - t[0]) - sigma**2
    ends = set_days[1:] + [n]
    p_path = p_step = 0.0
    for a, b in zip(set_days, ends):
        dollar_gamma = bs_gamma(S[a], K, expiry - t[a], r[a], q, sigma) * S[a] ** 2
        dt = t[b] - t[a]
        p_path += 0.5 * var_gap * dollar_gamma * dt
        p_step += 0.5 * dollar_gamma * ((S[b] / S[a] - 1.0) ** 2 - sigma**2 * dt)
    vega0 = bs_vega(S[0], K, expiry - t[0], r[0], q, sigma)
    p_gap = vega0 * var_gap / (2 * sigma)
    return {
        "pnl": pnl / c0,
        "payoff": payoff / c0,
        "stock": stock / c0,
        "financing": financing / c0,
        "costs": costs / c0,
        "turnover": turnover,
        "p_gap": p_gap / c0,
        "p_path": p_path / c0,
        "p_step": p_step / c0,
        "rv": rv,
        "c0": c0,
    }


@pytest.fixture(scope="module")
def uneven_window():
    """Five paths on an irregular calendar grid with daily rates and per-path inputs."""
    days = np.array([0, 1, 2, 3, 6, 7, 8, 9, 10, 13, 14])  # weekends skipped
    t = days / 365
    rng = np.random.default_rng(7)
    r = 0.03 + 0.01 * rng.standard_normal(t.size)
    sigma = np.array([0.12, 0.18, 0.25, 0.4, 0.18])
    S = gbm_paths(400.0, 0.03, sigma * 1.3, t, 5, seed=8)
    K = np.array([380.0, 400.0, 405.0, 420.0, 401.0])
    q = np.array([0.0, 0.013, 0.0, 0.02, 0.0])
    return S, t, r, sigma, K, q


@pytest.mark.parametrize("h", [1, 3, 4, 10, 12])
def test_matches_reference_loop(uneven_window, h):
    S, t, r, sigma, K, q = uneven_window
    c = 0.0025
    res = simulate_window(S, t, r, sigma, K, h=h, c=c, q=q)
    assert res.n_intervals == hedge_grid(len(t) - 1, h).size - 1
    for i in range(S.shape[0]):
        ref = _reference_window(S[i], t, r, sigma[i], K[i], h, c, q[i])
        for name, value in ref.items():
            assert getattr(res, name)[i] == pytest.approx(value, rel=1e-12, abs=1e-14), name


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


def test_static_hedge_has_no_interior_turnover(uneven_window):
    S, t, r, sigma, K, q = uneven_window
    res = simulate_window(S, t, r, sigma, K, h=len(t) - 1, c=0.001, q=q)
    assert res.n_intervals == 1
    np.testing.assert_array_equal(res.turnover, 0.0)


def test_rejects_non_increasing_times():
    with pytest.raises(ValueError):
        simulate_window([100.0, 101.0, 99.0], [0.0, 0.1, 0.1], 0.0, 0.2, 100.0, h=1, c=0.0)


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
    # DESIGN.md gives "about 0.252"; tolerance 0.005 is several Monte Carlo
    # standard errors yet excludes the Derman-Kamal value 0.273.
    res = simulate_window(flat_paths, TIMES, 0.0, SIGMA, S0, h=2, c=0.0)
    std = res.pnl.std()
    dk = _derman_kamal(21 / 2)  # asymptotic formula with interval 2/252
    print(f"every-2-days std P&L/C0 = {std:.5f}, Derman-Kamal = {dk:.6f}")
    assert dk == pytest.approx(0.273, abs=5e-4)
    assert std == pytest.approx(0.252, abs=0.005)


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


@pytest.mark.parametrize("q, expected", [(0.013, 0.0264), (0.0, 0.0)])
def test_carry_consistency(q, expected):
    # Total-return path with drift r; K is the pricing model's forward S0·exp((r - q)T).
    r = 0.04
    paths = gbm_paths(S0, r, SIGMA, TIMES, N_PATHS, seed=SEED_DRIFT)
    K = S0 * math.exp((r - q) * T)
    res = simulate_window(paths, TIMES, r, SIGMA, K, h=1, c=0.0, q=q)
    mean = res.pnl.mean()
    print(f"carry q = {q}: mean P&L/C0 = {mean:.5f}")
    assert mean == pytest.approx(expected, abs=0.003)


def test_p_step_r2_on_synthetic_window_set():
    # DESIGN.md section 3: 3,000 windows, sigma_i ~ lognormal(ln 0.17, 0.35),
    # sigma_true = sigma_i·exp(N(-0.2, 0.3)), daily hedging, r = q = 0, K = F0.
    rng = np.random.default_rng(SEED_SET_VOLS)
    sigma_i = np.exp(rng.normal(math.log(0.17), 0.35, 3000))
    sigma_true = sigma_i * np.exp(rng.normal(-0.2, 0.3, 3000))
    paths = gbm_paths(S0, 0.0, sigma_true, TIMES, 3000, seed=SEED_SET_PATHS)
    res = simulate_window(paths, TIMES, 0.0, sigma_i, S0, h=1, c=0.0)
    ladder = [r2_45(res.pnl, p) for p in (res.p_gap, res.p_path, res.p_step)]
    print("daily R2_45 P_gap / P_path / P_step = " + " / ".join(f"{x:.4f}" for x in ladder))
    assert ladder[2] >= 0.95
