"""Delta-hedging simulation and P&L attribution (DESIGN.md section 3).

A window is a path of closes S_0, ..., S_n at times t_0 < ... < t_n (years,
ACT/365). At the t_0 close the trader buys one European call or put struck at
K at its Black-Scholes value with fixed implied vol sigma_i, and holds -Delta
shares: the hedge is set at the t_0 close and at every h-th close counted back
from t_n, and unwound at the t_n close, when the option pays its payoff. All
cash (premium, stock trades, costs) sits in one account; the step rate r_j
prices and hedges at close j and accrues cash from close j to close j + 1.

Money units: every quantity is measured at the t_n close (expiry money) and
divided by C0·G_0, the premium carried to expiry, where G_j is the growth of
one unit of cash from close j to close n at the step rates. Predictor terms of
the hedge interval that starts at close k are weighted by G_k. The P&L is also
split by hedge interval, marking the option at its Black-Scholes value on each
grid close, and the interval P&Ls sum to the window P&L exactly.

The path is treated as a total-return series: no dividend cash flows are
credited or charged. A dividend yield q, if given, enters pricing and Greeks
only; Stage 5 uses q = 0 because yfinance adjusted closes already contain
dividends.

Shapes: one window has S and t of shape (n + 1,) and r of shape (n,); P
windows of the same length stack along a leading axis as (P, n + 1) and
(P, n). Windows of different lengths go through simulate_windows, and GBM
paths on the calendars of real windows through simulate_gbm_windows.

The real-data study adds the rolling windows (study_windows), the April 2025
exclusion (windows_containing), non-overlapping subsamples, the R² ladder with
moving block bootstrap intervals, hedging error statistics and the weekday
diagnostic of P_step residuals.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from volsurf.black_scholes import bs_delta, bs_gamma, bs_price, bs_vega


class HedgeResult(NamedTuple):
    """Output of simulate_window and simulate_windows.

    P&L quantities are in expiry money as fractions of C0·G_0 (the premium
    carried to expiry at the step rates), and
    pnl = payoff + premium + stock + financing + costs exactly.

    Attributes
    ----------
    pnl : ndarray
        Final value of the hedged position.
    payoff : ndarray
        Option payoff, max(S_n - K, 0) for a call or max(K - S_n, 0) for a put.
    premium : ndarray
        Premium paid at t_0 and carried to expiry, -C0·G_0 (so -1).
    stock : ndarray
        Price P&L of the stock position, -sum_k Delta_k·(S_{k+1} - S_k).
    financing : ndarray
        Interest earned (negative if paid) on the stock-trade and cost cash
        flows; the premium's interest is inside premium.
    costs : ndarray
        Transaction costs, -c times notional traded, on every trade.
    turnover : ndarray
        Interior turnover sum |Delta_k - Delta_{k-1}| over the rebalances
        strictly between the opening and closing trades, in shares per
        option (not a fraction of C0).
    p_gap : ndarray
        Vol-gap predictor G_0·vega_0·(sigma_r² - sigma_i²)/(2·sigma_i).
    p_path : ndarray
        Gamma-path predictor ½·(sigma_r² - sigma_i²)·sum_k G_k·Gamma_k·S_k²·dt_k.
    p_step : ndarray
        Step predictor sum_k ½·G_k·Gamma_k·S_k²·(R_k² - sigma_i²·dt_k).
    interval_pnl : ndarray, trailing axis of length N
        Hedged P&L of each hedge interval a -> b:
        G_b·V_b - G_a·V_a - Delta_a·(G_b·S_b - G_a·S_a) + G_a·cost_a, where V
        is the Black-Scholes value at sigma_i, r_a and the remaining time
        (V_n is the payoff) and cost_a <= 0 is the cost of the trade at a; the
        last interval also carries the unwind cost. Sums to pnl.
    interval_p_step : ndarray, trailing axis of length N
        The P_step term of each interval, ½·G_a·Gamma_a·S_a²·(R² - sigma_i²·dt).
        Sums to p_step.
    rv : ndarray
        Realised total variance sum_j ln(S_{j+1}/S_j)² over daily closes
        (not annualised).
    c0 : ndarray
        Initial premium C0 in currency units at t_0.
    n_intervals : int or ndarray of int
        Number of hedge intervals N in the window (an array with one entry
        per window from simulate_windows).
    """

    pnl: np.ndarray
    payoff: np.ndarray
    premium: np.ndarray
    stock: np.ndarray
    financing: np.ndarray
    costs: np.ndarray
    turnover: np.ndarray
    p_gap: np.ndarray
    p_path: np.ndarray
    p_step: np.ndarray
    interval_pnl: np.ndarray
    interval_p_step: np.ndarray
    rv: np.ndarray
    c0: np.ndarray
    n_intervals: int | np.ndarray


_INTERVAL_FIELDS = ("interval_pnl", "interval_p_step")
_FIELDS = tuple(f for f in HedgeResult._fields if f not in _INTERVAL_FIELDS + ("n_intervals",))


def gbm_paths(S0, mu, sigma, t, n_paths, seed):
    """Exact geometric Brownian motion paths on a fixed time grid.

    S_{j+1} = S_j·exp((mu - sigma²/2)·dt_j + sigma·sqrt(dt_j)·Z_j), Z_j iid N(0, 1).

    Parameters
    ----------
    S0 : float
        Initial price.
    mu : float or array_like of shape (n,) or (n_paths, n)
        Drift per year as a decimal (arithmetic, so E[S_t] = S0·exp(mu·t)
        for a constant mu); an array gives each step, and optionally each
        path, its own drift.
    sigma : float or array_like of shape (n_paths,)
        Volatility per year as a decimal; an array gives each path its own.
    t : array_like of shape (n + 1,) or (n_paths, n + 1)
        Increasing times in years; t[..., 0] is the time of S0. A 2-D grid
        gives each path its own calendar.
    n_paths : int
        Number of paths.
    seed : int or sequence of int
        Seed for np.random.default_rng.

    Returns
    -------
    ndarray of shape (n_paths, n + 1)
        Prices, with column 0 equal to S0.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    dt = np.diff(t, axis=-1)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)[..., None]
    z = rng.standard_normal((n_paths, dt.shape[-1]))
    log_steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=-1)], axis=-1)
    return S0 * np.exp(log_path)


def hedge_grid(n_steps, h):
    """Indices of the closes where the hedge is set, plus the final close.

    The grid is anchored to expiry: every interval has length h except the
    first, which is shorter when h does not divide n. A stub therefore opens
    the window, where gamma is low, rather than sitting on the final days,
    where ATM gamma peaks.

    Parameters
    ----------
    n_steps : int
        Number of daily steps n in the window (closes 0, ..., n).
    h : int
        Rebalance interval in closes, h >= 1.

    Returns
    -------
    ndarray of int
        [0, ..., n - 2h, n - h, n], holding 0 and every n - m·h above 0. Its
        length is N + 1, where N = ceil(n/h) is the number of hedge
        intervals.
    """
    if n_steps < 1:
        raise ValueError("a window needs at least one step")
    if h < 1:
        raise ValueError("rebalance interval h must be at least 1")
    return np.append(0, n_steps - np.arange(0, n_steps, h)[::-1])


def r2_45(y, p):
    """R² of predictions about the 45 degree line, along the last axis.

    Parameters
    ----------
    y, p : array_like of the same shape (..., n)
        Outcomes and predictions (same units); leading axes index separate
        samples, for example bootstrap resamples.

    Returns
    -------
    float or ndarray of shape (...)
        1 - sum (y - p)² / sum (y - mean(y))². Equals 1 for perfect
        predictions and 0 when p is the sample mean; negative when p does
        worse than the mean. Unlike the OLS R², it penalises any intercept
        or slope away from 1.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    spread = y - y.mean(axis=-1, keepdims=True)
    return 1.0 - np.sum((y - p) ** 2, axis=-1) / np.sum(spread**2, axis=-1)


def ols_line(y, p):
    """Ordinary least squares fit y = alpha + beta·p, along the last axis.

    Parameters
    ----------
    y, p : array_like of the same shape (..., n)
        Outcomes and predictions (same units).

    Returns
    -------
    alpha, beta : float or ndarray of shape (...)
        Intercept (units of y) and slope. Theory for a perfect predictor
        gives alpha = 0 and beta = 1.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    y_bar = y.mean(axis=-1, keepdims=True)
    p_bar = p.mean(axis=-1, keepdims=True)
    beta = np.sum((p - p_bar) * (y - y_bar), axis=-1) / np.sum((p - p_bar) ** 2, axis=-1)
    alpha = y_bar[..., 0] - beta * p_bar[..., 0]
    return alpha, beta


def _check_shape(x, name, allowed):
    """Raise ValueError unless x.shape is one of the allowed shapes."""
    allowed = list(dict.fromkeys(allowed))
    if x.shape not in allowed:
        options = " or ".join(str(s) for s in allowed)
        raise ValueError(f"{name} must have shape {options}; got {x.shape}")


def _validate_inputs(S, t, r, sigma_i, K, q):
    """Cast the inputs of simulate_window to float arrays and check their shapes.

    S fixes the layout: (n + 1,) for one window or (P, n + 1) for P windows.
    t is (n + 1,) or (P, n + 1); r is a scalar, (n,) or (P, n); sigma_i, K
    and q are scalars or (P,), and must be scalars when S is 1-D. Any other
    shape raises ValueError, so a path axis can never be read as a time axis.

    Returns
    -------
    S, t, r, sigma_i, K, q : ndarray
        S and t of shape lead + (n + 1,), r of shape lead + (n,) and the
        per-window inputs of shape lead, where lead is () or (P,).
    """
    S, t, r, sigma_i, K, q = (np.asarray(x, dtype=float) for x in (S, t, r, sigma_i, K, q))
    if S.ndim not in (1, 2) or S.shape[-1] < 2:
        raise ValueError(f"S must have shape (n + 1,) or (paths, n + 1) with n >= 1; got {S.shape}")
    lead, n = S.shape[:-1], S.shape[-1] - 1
    _check_shape(t, "t", [(n + 1,), lead + (n + 1,)])
    _check_shape(r, "r", [(), (n,), lead + (n,)])
    for x, name in ((sigma_i, "sigma_i"), (K, "K"), (q, "q")):
        _check_shape(x, name, [(), lead])
    t = np.broadcast_to(t, S.shape)
    r = np.broadcast_to(r, lead + (n,))
    sigma_i, K, q = (np.broadcast_to(x, lead) for x in (sigma_i, K, q))
    return S, t, r, sigma_i, K, q


def simulate_window(S, t, r, sigma_i, K, h, c, q=0.0, is_call=True):
    """Delta-hedge a long European option along price paths of one length.

    Parameters
    ----------
    S : array_like of shape (n + 1,) or (P, n + 1)
        Closes S_0, ..., S_n (n >= 1) of one window, or of P windows.
    t : array_like of shape (n + 1,) or (P, n + 1)
        Increasing times of the closes in years (ACT/365), shared or one row
        per window; only differences matter. The option expires at t_n, so
        T0 = t_n - t_0.
    r : float or array_like of shape (n,) or (P, n)
        Continuously compounded step rates. r_j prices and hedges at close j
        and accrues cash from close j to close j + 1; a scalar is a flat rate.
    sigma_i : float or array_like of shape (P,)
        Implied vol as a decimal, fixed for the window.
    K : float or array_like of shape (P,)
        Strike.
    h : int
        Rebalance interval in closes: the hedge is set at close 0 and at
        closes n - h, n - 2h, ... above 0 (see hedge_grid), and unwound at
        close n.
    c : float
        Proportional cost as a decimal fraction of notional traded
        (5 bp = 0.0005), charged on every trade including the opening and
        closing ones.
    q : float or array_like of shape (P,), default 0
        Dividend yield used in pricing and Greeks only (see module docstring).
    is_call : bool, default True
        True for a call, False for a put.

    Returns
    -------
    HedgeResult
        P&L, its components and the three predictors, each in expiry money
        as a fraction of C0·G_0 with shape () or (P,), plus turnover, rv, c0
        and N. interval_pnl and interval_p_step have shape () + (N,) or
        (P, N), interval k running from grid close k to grid close k + 1 of
        hedge_grid(n, h). Delta, Gamma and the option mark at close k use
        sigma_i, rate r_k and remaining time t_n - t_k; R_k = S_{k+1}/S_k - 1
        and dt_k = t_{k+1} - t_k run over the hedge grid, and
        sigma_r² = rv/T0.

    Raises
    ------
    ValueError
        If a shape is not one listed above, or the times do not increase.
    """
    S, t, r, sigma_i, K, q = _validate_inputs(S, t, r, sigma_i, K, q)
    if np.any(np.diff(t, axis=-1) <= 0):
        raise ValueError("times must be strictly increasing")

    lead = S.shape[:-1]
    n = S.shape[-1] - 1
    grid = hedge_grid(n, h)
    set_closes = grid[:-1]
    n_intervals = grid.size - 1
    tau = t[..., -1:] - t  # remaining time at each close
    T0 = tau[..., 0]
    sig, strike, div = sigma_i[..., None], K[..., None], q[..., None]

    # Hedge grid: closes where the hedge is set (all but the last grid point).
    S_g, t_g = S[..., grid], t[..., grid]
    S_set, tau_set, r_set = S_g[..., :-1], tau[..., set_closes], r[..., set_closes]
    delta = bs_delta(S_set, strike, tau_set, r_set, div, sig, is_call)
    gamma = bs_gamma(S_set, strike, tau_set, r_set, div, sig)
    value = bs_price(S_set, strike, tau_set, r_set, div, sig, is_call)  # option marks; value[..., 0] is C0
    c0 = bs_price(S[..., 0], K, T0, r[..., 0], q, sigma_i, is_call)
    vega0 = bs_vega(S[..., 0], K, T0, r[..., 0], q, sigma_i)

    # Growth G_j of one unit of cash from close j to close n; C0·G_0 is the unit.
    log_growth = r * np.diff(t, axis=-1)
    tail = np.cumsum(log_growth[..., ::-1], axis=-1)[..., ::-1]
    growth = np.exp(np.concatenate([tail, np.zeros(lead + (1,))], axis=-1))
    g0, g_set, g_grid = growth[..., 0], growth[..., set_closes], growth[..., grid]
    unit = c0 * g0

    # Stock position -Delta_k after each grid close, 0 after the unwind.
    position = np.concatenate([-delta, np.zeros(lead + (1,))], axis=-1)
    trade = np.diff(position, axis=-1, prepend=0.0)  # shares bought at each grid close
    cost_flow = -c * np.abs(trade) * S_g
    omega = 1.0 if is_call else -1.0
    payoff = np.maximum(omega * (S[..., -1] - K), 0.0)
    stock = -np.sum(delta * np.diff(S_g, axis=-1), axis=-1)  # = -sum(trade·S_g)
    costs = np.sum(cost_flow, axis=-1)
    flow = -trade * S_g + cost_flow  # cash in at each grid close
    financing = np.sum(flow * (g_grid - 1.0), axis=-1)

    pnl = payoff - unit + stock + financing + costs
    turnover = np.sum(np.abs(trade[..., 1:-1]), axis=-1)

    # P&L by interval in expiry money. Cash held over an interval keeps its expiry value, so
    # only the option mark, the hedge and the cost of the opening trade change it; the last
    # interval also pays the unwind cost. The marks telescope from C0·G_0 to the payoff.
    value_end = np.concatenate([value[..., 1:], payoff[..., None]], axis=-1)
    carried_cost = cost_flow * g_grid
    interval_pnl = (
        g_grid[..., 1:] * value_end
        - g_set * value
        - delta * (g_grid[..., 1:] * S_g[..., 1:] - g_set * S_set)
        + carried_cost[..., :-1]
    )
    interval_pnl[..., -1] += carried_cost[..., -1]

    # Predictors on the hedge grid, each term carried to expiry.
    rv = np.sum(np.diff(np.log(S), axis=-1) ** 2, axis=-1)
    var_gap = rv / T0 - sigma_i**2
    dt_g = np.diff(t_g, axis=-1)
    ret_g = S_g[..., 1:] / S_g[..., :-1] - 1.0
    dollar_gamma = g_set * gamma * S_set**2
    p_gap = g0 * vega0 * var_gap / (2.0 * sigma_i)
    p_path = 0.5 * var_gap * np.sum(dollar_gamma * dt_g, axis=-1)
    interval_p_step = 0.5 * dollar_gamma * (ret_g**2 - sig**2 * dt_g)
    p_step = np.sum(interval_p_step, axis=-1)
    unit_k = np.asarray(unit)[..., None]

    return HedgeResult(
        pnl=pnl / unit,
        payoff=payoff / unit,
        premium=-np.ones(lead),
        stock=stock / unit,
        financing=financing / unit,
        costs=costs / unit,
        turnover=turnover,
        p_gap=p_gap / unit,
        p_path=p_path / unit,
        p_step=p_step / unit,
        interval_pnl=interval_pnl / unit_k,
        interval_p_step=interval_p_step / unit_k,
        rv=rv,
        c0=c0,
        n_intervals=n_intervals,
    )


def simulate_windows(S, t, r, starts, ends, sigma_i, K, h, c, q=0.0, is_call=True):
    """Delta-hedge windows of one history, calling simulate_window once per length.

    Real windows span a fixed number of calendar days, so their number of
    trading-day steps n varies. Windows are grouped by n, each group is
    stacked into (P, n + 1) arrays with its own calendar and rates, and the
    results are returned in the order the windows were given.

    Parameters
    ----------
    S, t, r : array_like of shape (M,)
        History of closes, their times in years (ACT/365) and the
        continuously compounded rate at each close.
    starts, ends : array_like of int, shape (W,)
        Window w runs from close starts[w] to close ends[w], with
        0 <= starts[w] < ends[w] < M, so it has n_w = ends[w] - starts[w]
        steps and step rates r[starts[w]:ends[w]].
    sigma_i, K : float or array_like of shape (W,)
        Implied vol (decimal) and strike of each window.
    h, c, is_call
        As in simulate_window.
    q : float or array_like of shape (W,), default 0
        As in simulate_window.

    Returns
    -------
    HedgeResult
        Every field has shape (W,) in window order and n_intervals is an int
        array, except interval_pnl and interval_p_step, which have shape
        (W, N_max) with N_max the largest N. They are aligned at expiry:
        column j is the interval ending at close ends[w] - (N_max - 1 - j)·h
        of the history, and columns before a window's first interval are
        NaN.

    Raises
    ------
    ValueError
        If the history, the bounds or a per-window input has the wrong shape
        or type, or a window is empty or out of range.
    """
    S, t, r = _histories(S, t, r)
    starts, ends = _window_bounds(starts, ends, S.size)
    n_windows = starts.size
    sigma_i, K, q = (_per_window(x, name, n_windows) for x, name in ((sigma_i, "sigma_i"), (K, "K"), (q, "q")))

    lengths = ends - starts
    parts = []
    for n in np.unique(lengths):
        rows = np.flatnonzero(lengths == n)
        closes = starts[rows, None] + np.arange(n + 1)
        res = simulate_window(
            S[closes], t[closes], r[closes[:, :-1]], sigma_i[rows], K[rows], h, c, q[rows], is_call
        )
        parts.append((rows, res))
    return _assemble(parts, n_windows)


def _histories(*arrays):
    """Cast histories to float arrays; raise unless all are 1-D of one length."""
    arrays = [np.asarray(x, dtype=float) for x in arrays]
    if arrays[0].ndim != 1 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("histories must be 1-D arrays of the same length")
    return arrays


def _window_bounds(starts, ends, size):
    """Check window bounds into a history of length size and return them as arrays."""
    starts, ends = np.asarray(starts), np.asarray(ends)
    if starts.ndim != 1 or ends.shape != starts.shape:
        raise ValueError("starts and ends must be 1-D arrays of the same length")
    if not (np.issubdtype(starts.dtype, np.integer) and np.issubdtype(ends.dtype, np.integer)):
        raise ValueError("starts and ends must be integer indices")
    if np.any(starts < 0) or np.any(ends <= starts) or np.any(ends >= size):
        raise ValueError("every window needs 0 <= start < end < len(S)")
    return starts, ends


def _per_window(x, name, n_windows):
    """Broadcast a scalar or (W,) per-window input to shape (W,); raise otherwise."""
    x = np.asarray(x, dtype=float)
    _check_shape(x, name, [(), (n_windows,)])
    return np.broadcast_to(x, (n_windows,))


def _assemble(parts, n_windows, extra=()):
    """Gather results of window groups into window order.

    Parameters
    ----------
    parts : list of (rows, HedgeResult)
        rows indexes the windows of one group. Its result has per-window
        fields of shape (len(rows),) + extra, interval fields of shape
        (len(rows),) + extra + (N,), and an int n_intervals.
    n_windows : int
        Total number of windows W.
    extra : tuple of int, default ()
        Trailing shape of each window's results (for example paths per
        window).

    Returns
    -------
    HedgeResult
        Per-window fields of shape (W,) + extra; interval fields of shape
        (W,) + extra + (N_max,), aligned at expiry and NaN-padded on the
        left; n_intervals an int array of shape (W,).
    """
    n_max = max(res.n_intervals for _, res in parts)
    out = {name: np.empty((n_windows,) + extra) for name in _FIELDS}
    out.update({name: np.full((n_windows,) + extra + (n_max,), np.nan) for name in _INTERVAL_FIELDS})
    n_intervals = np.empty(n_windows, dtype=int)
    for rows, res in parts:
        for name in _FIELDS:
            out[name][rows] = getattr(res, name)
        for name in _INTERVAL_FIELDS:
            out[name][rows, ..., n_max - res.n_intervals :] = getattr(res, name)
        n_intervals[rows] = res.n_intervals
    return HedgeResult(**out, n_intervals=n_intervals)


def simulate_gbm_windows(t, r, starts, ends, sigma_true, sigma_i, h, c, n_paths, seed):
    """Delta-hedge GBM paths on the calendars and rates of real windows.

    Window w gets n_paths paths that start at 100 on the closes
    t[starts[w]], ..., t[ends[w]], with volatility sigma_true[w] and, at each
    step, drift equal to the step rate (the risk-neutral drift of a
    total-return series). Each path is hedged as the real window is: implied
    vol sigma_i[w], financing at the same step rates, strike
    F0 = 100·exp(r0·T0) and q = 0. P&L fractions do not depend on the price
    level, so starting every path at 100 loses nothing. Windows of n steps
    draw their normals from np.random.default_rng([seed, n]), so one seed
    gives the same normals whatever sigma_true, h or c.

    Parameters
    ----------
    t, r : array_like of shape (M,)
        Times of the history's closes in years (ACT/365) and the
        continuously compounded rate at each close.
    starts, ends : array_like of int, shape (W,)
        Window bounds, as in simulate_windows.
    sigma_true, sigma_i : float or array_like of shape (W,)
        Volatility of the simulated paths and implied vol used to price and
        hedge, as decimals.
    h, c
        As in simulate_window.
    n_paths : int
        Paths per window.
    seed : int
        Base seed.

    Returns
    -------
    HedgeResult
        Per-path fields of shape (W, n_paths), interval fields of shape
        (W, n_paths, N_max) aligned at expiry as in simulate_windows, and
        n_intervals of shape (W,).
    """
    t, r = _histories(t, r)
    starts, ends = _window_bounds(starts, ends, t.size)
    n_windows = starts.size
    sigma_true = _per_window(sigma_true, "sigma_true", n_windows)
    sigma_i = _per_window(sigma_i, "sigma_i", n_windows)

    lengths = ends - starts
    parts = []
    for n in np.unique(lengths):
        rows = np.flatnonzero(lengths == n)
        closes = starts[rows, None] + np.arange(n + 1)
        # Window-major rows: the n_paths paths of one window are adjacent.
        times = np.repeat(t[closes], n_paths, axis=0)
        rates = np.repeat(r[closes[:, :-1]], n_paths, axis=0)
        vols = np.repeat(sigma_true[rows], n_paths)
        paths = gbm_paths(100.0, rates, vols, times, times.shape[0], seed=[seed, int(n)])
        strike = 100.0 * np.exp(rates[:, 0] * (times[:, -1] - times[:, 0]))
        res = simulate_window(paths, times, rates, np.repeat(sigma_i[rows], n_paths), strike, h, c)
        by_window = {
            name: getattr(res, name).reshape((rows.size, n_paths) + getattr(res, name).shape[1:])
            for name in _FIELDS + _INTERVAL_FIELDS
        }
        parts.append((rows, res._replace(**by_window)))
    return _assemble(parts, n_windows, extra=(n_paths,))


def synthetic_window_set(n_windows, seed_vols, seed_paths):
    """The synthetic GBM window set of DESIGN.md section 3.

    sigma_i ~ lognormal(ln 0.17, 0.35), then sigma_true = sigma_i·exp(N(-0.2, 0.3)),
    both from np.random.default_rng(seed_vols). Paths start at 100 and run
    21 daily steps of 1/252 years with zero drift and volatility sigma_true
    (gbm_paths with seed_paths). Hedge them with r = q = 0 and K = F0 = 100.

    Parameters
    ----------
    n_windows : int
        Number of windows.
    seed_vols, seed_paths : int
        Seeds for the volatility draws and for the paths.

    Returns
    -------
    S : ndarray of shape (n_windows, 22)
        Prices.
    t : ndarray of shape (22,)
        Times in years.
    sigma_i : ndarray of shape (n_windows,)
        Implied vols as decimals.
    """
    rng = np.random.default_rng(seed_vols)
    sigma_i = np.exp(rng.normal(np.log(0.17), 0.35, n_windows))
    sigma_true = sigma_i * np.exp(rng.normal(-0.2, 0.3, n_windows))
    t = np.arange(22) / 252
    return gbm_paths(100.0, 0.0, sigma_true, t, n_windows, seed=seed_paths), t, sigma_i


# ---------------------------------------------------------------------------
# Real-data study (DESIGN.md section 3)
# ---------------------------------------------------------------------------


class StudyWindows(NamedTuple):
    """Rolling windows of the real-data study, as built by study_windows.

    Attributes
    ----------
    t : ndarray of shape (M,)
        Time of every close in years since the first close (ACT/365).
    starts, ends : ndarray of int, shape (W,)
        History indices of each window's t0 and t_end closes.
    T0 : ndarray of shape (W,)
        Window length t_end - t0 in years.
    sigma_i : ndarray of shape (W,)
        Implied vol at t0 as a decimal (^VIX close / 100).
    K : ndarray of shape (W,)
        Strike F0 = S0·exp(r0·T0).
    """

    t: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    T0: np.ndarray
    sigma_i: np.ndarray
    K: np.ndarray


def study_windows(history, first_start="2021-10-01", horizon_days=30):
    """Rolling windows of DESIGN.md section 3, one per trading day.

    Windows start on every trading day t0 from the first one on or after
    first_start. t_end is the last trading day on or before t0 plus
    horizon_days calendar days, and a window is kept only when t0 plus
    horizon_days is on or before the last date of the history, so every
    window is complete.

    Parameters
    ----------
    history : pd.DataFrame
        As returned by data.load_history: unique, increasing trading dates
        as the index, and columns spy (price), sigma_i (decimal) and r
        (continuously compounded decimal).
    first_start : str or pd.Timestamp, default "2021-10-01"
        Earliest start date.
    horizon_days : int, default 30
        Calendar days from t0 to the nominal expiry.

    Returns
    -------
    StudyWindows
        Times, bounds, lengths, implied vols and strikes of the windows.

    Raises
    ------
    ValueError
        If the dates are not unique and increasing.
    """
    dates = pd.DatetimeIndex(history.index)
    if not (dates.is_unique and dates.is_monotonic_increasing):
        raise ValueError("history dates must be unique and increasing")
    t = np.asarray((dates - dates[0]).days, dtype=float) / 365
    horizon = dates + pd.Timedelta(days=horizon_days)
    starts = np.arange(dates.searchsorted(pd.Timestamp(first_start)), dates.size)
    starts = starts[horizon[starts] <= dates[-1]]
    ends = dates.searchsorted(horizon[starts], side="right") - 1
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]

    S = history["spy"].to_numpy(dtype=float)
    r = history["r"].to_numpy(dtype=float)
    T0 = t[ends] - t[starts]
    K = S[starts] * np.exp(r[starts] * T0)
    sigma_i = history["sigma_i"].to_numpy(dtype=float)[starts]
    return StudyWindows(t=t, starts=starts, ends=ends, T0=T0, sigma_i=sigma_i, K=K)


def windows_containing(dates, starts, ends, first, last):
    """Flag windows whose closes include any trading day from first to last.

    Parameters
    ----------
    dates : DatetimeIndex or array_like of datetime64, shape (M,)
        Increasing dates of the history's closes.
    starts, ends : array_like of int, shape (W,)
        History indices of each window's first and last close.
    first, last : str or pd.Timestamp
        Inclusive date range.

    Returns
    -------
    ndarray of bool, shape (W,)
        True where some close starts[w], ..., ends[w] falls in the range.
    """
    dates = pd.DatetimeIndex(dates)
    lo = dates.searchsorted(pd.Timestamp(first), side="left")
    hi = dates.searchsorted(pd.Timestamp(last), side="right") - 1
    starts, ends = np.asarray(starts), np.asarray(ends)
    return (lo <= hi) & (starts <= hi) & (ends >= lo)


def non_overlapping(starts, ends, first=0):
    """Greedy subsample of windows that share no daily return.

    Starting from window first, each next window is the first, in start
    order, whose start close is at or after the previous window's end close.

    Parameters
    ----------
    starts, ends : array_like of int, shape (W,)
        History indices of each window's first and last close; starts must
        be strictly increasing.
    first : int, default 0
        Index of the window the subsample starts from.

    Returns
    -------
    ndarray of int
        Increasing window indices, beginning with first.

    Raises
    ------
    ValueError
        If starts is not strictly increasing or first is out of range.
    """
    starts, ends = np.asarray(starts), np.asarray(ends)
    if np.any(np.diff(starts) <= 0):
        raise ValueError("starts must be strictly increasing")
    if not 0 <= first < starts.size:
        raise ValueError("first must index a window")
    chosen = [first]
    while (nxt := np.searchsorted(starts, ends[chosen[-1]], side="left")) < starts.size:
        chosen.append(int(nxt))
    return np.array(chosen)


def block_bootstrap_indices(n, block, n_resamples, seed):
    """Moving block bootstrap resamples of a sequence of n observations.

    Each resample joins ceil(n/block) blocks of block consecutive indices,
    whose first indices are drawn uniformly, with replacement, from
    0, ..., n - block, and keeps the first n indices.

    Parameters
    ----------
    n : int
        Sequence length.
    block : int
        Block length, 1 <= block <= n.
    n_resamples : int
        Number of resamples.
    seed : int
        Seed for np.random.default_rng.

    Returns
    -------
    ndarray of int, shape (n_resamples, n)
        Indices into the sequence, one resample per row.
    """
    if not 1 <= block <= n:
        raise ValueError("block length must be between 1 and n")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    first = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    return (first[..., None] + np.arange(block)).reshape(n_resamples, -1)[:, :n]


_LADDER_COLUMNS = ["n", "r2_45", "r2_lo", "r2_hi", "alpha", "beta", "beta_lo", "beta_hi"]


def r2_ladder(y, predictors, boot_idx=None, level=0.95):
    """R² about the 45 degree line and the OLS line of each predictor, with bootstrap intervals.

    Parameters
    ----------
    y : array_like of shape (n,)
        Outcomes, for example hedged P&L as a fraction of C0·G0.
    predictors : dict of str to array_like of shape (n,)
        Predictions in the units of y, in the order of the output rows.
    boot_idx : ndarray of int, shape (B, n), optional
        Resampled indices (block_bootstrap_indices). Every predictor uses
        the same resamples, and each statistic is recomputed on each.
    level : float, default 0.95
        Coverage of the percentile intervals.

    Returns
    -------
    pd.DataFrame
        One row per predictor with columns n, r2_45 (r2_45), r2_lo and r2_hi
        (its interval), alpha and beta (ols_line; alpha in the units of y),
        beta_lo and beta_hi. Interval columns are NaN without boot_idx.
    """
    y = np.asarray(y, dtype=float)
    tails = [50.0 * (1.0 - level), 100.0 - 50.0 * (1.0 - level)]
    rows = {}
    for name, p in predictors.items():
        p = np.asarray(p, dtype=float)
        if y.ndim != 1 or p.shape != y.shape:
            raise ValueError(f"y and {name} must be 1-D arrays of the same length")
        alpha, beta = ols_line(y, p)
        row = {"n": y.size, "r2_45": r2_45(y, p), "alpha": alpha, "beta": beta}
        row.update(r2_lo=np.nan, r2_hi=np.nan, beta_lo=np.nan, beta_hi=np.nan)
        if boot_idx is not None:
            y_b, p_b = y[boot_idx], p[boot_idx]
            row["r2_lo"], row["r2_hi"] = np.percentile(r2_45(y_b, p_b), tails)
            row["beta_lo"], row["beta_hi"] = np.percentile(ols_line(y_b, p_b)[1], tails)
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")[_LADDER_COLUMNS]


def hedging_error_stats(pnl, p_gap):
    """Mean, std and RMSE of the hedging error e = pnl - p_gap.

    Parameters
    ----------
    pnl, p_gap : array_like of the same shape
        Hedged P&L and vol-gap predictor as fractions of C0·G0, over windows
        (and paths); every element counts once.

    Returns
    -------
    dict
        mean, std and rmse of e as fractions of C0·G0. std is the population
        std (ddof 0), so rmse = √(mean² + std²) = √mean(e²).
    """
    e = np.ravel(np.asarray(pnl, dtype=float) - np.asarray(p_gap, dtype=float))
    mean, std = e.mean(), e.std()
    return {"mean": mean, "std": std, "rmse": np.sqrt(mean**2 + std**2)}


def derman_kamal(n_intervals, vega0, sigma_i, c0):
    """Asymptotic std of discretely delta-hedged P&L with known vol (Derman-Kamal).

    Parameters
    ----------
    n_intervals : float or array_like
        Number of equal hedge intervals N.
    vega0 : float or array_like
        Vega at t0 per unit of volatility.
    sigma_i : float or array_like
        Implied (and true) vol as a decimal.
    c0 : float or array_like
        Premium, in the currency units of vega0.

    Returns
    -------
    float or ndarray
        √(π/4)·vega0·sigma_i/(√N·c0), as a fraction of the premium.
    """
    return np.sqrt(np.pi / 4) * vega0 * sigma_i / (np.sqrt(n_intervals) * c0)


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_WEEKDAY_COLUMNS = [
    "intervals", "calendar_days", "pnl", "p_step", "residual", "residual_lo", "residual_hi", "share",
]


def weekday_diagnostic(interval_pnl, interval_p_step, dates, starts, ends, h, boot_idx=None, level=0.95):
    """P_step residuals grouped by the weekday of each hedge interval's closing day.

    The residual of an interval is its hedged P&L minus its P_step term.
    Grouping by the closing weekday separates Friday-to-Monday intervals,
    which charge three calendar days of sigma_i² against about one trading
    day of realised variance.

    Parameters
    ----------
    interval_pnl, interval_p_step : array_like of shape (W, N)
        As returned by simulate_windows: aligned at expiry, NaN before a
        window's first interval, fractions of C0·G0.
    dates : DatetimeIndex or array_like of datetime64, shape (M,)
        Dates of the history's closes.
    starts, ends : array_like of int, shape (W,)
        Window bounds passed to simulate_windows.
    h : int
        Rebalance interval passed to simulate_windows.
    boot_idx : ndarray of int, shape (B, W), optional
        Window resamples (block_bootstrap_indices); the mean residuals are
        recomputed on each resample for percentile intervals.
    level : float, default 0.95
        Coverage of the intervals.

    Returns
    -------
    table : pd.DataFrame
        One row per weekday, Monday to Friday, and a final row All. Columns:
        intervals (count), calendar_days (mean interval length in calendar
        days), pnl, p_step and residual (means per interval, fractions of
        C0·G0), residual_lo and residual_hi (interval for the mean residual,
        NaN without boot_idx) and share (the weekday's sum of residuals over
        the sum of all residuals).
    eta2 : float
        Share of the residuals' variance explained by the weekday means,
        sum_d n_d·(mean_d - mean)² / sum (residual - mean)².
    """
    pnl = np.asarray(interval_pnl, dtype=float)
    step = np.asarray(interval_p_step, dtype=float)
    dates = pd.DatetimeIndex(dates)
    starts, ends = np.asarray(starts), np.asarray(ends)
    valid = ~np.isnan(pnl)

    # Closes bounding each interval; padded cells point at t_end and are masked out.
    end_close = ends[:, None] - h * np.arange(pnl.shape[1])[::-1]
    start_close = np.maximum(end_close - h, starts[:, None])
    end_close = np.where(valid, end_close, ends[:, None])
    start_close = np.where(valid, start_close, ends[:, None])
    day = dates.to_numpy()
    days = (day[end_close] - day[start_close]) / np.timedelta64(1, "D")
    weekday = np.where(valid, dates.weekday.to_numpy()[end_close], -1)
    member = weekday[..., None] == np.arange(len(_WEEKDAYS))  # (W, N, 5); False on padding

    def by_weekday(x):
        """Per-window sums of x over the intervals closing on each weekday, shape (W, 5)."""
        return np.sum(np.where(member, x[..., None], 0.0), axis=1)

    resid = pnl - step
    counts, resid_w = member.sum(axis=1), by_weekday(resid)
    n_d = counts.sum(axis=0)
    table = pd.DataFrame(index=_WEEKDAYS + ["All"], columns=_WEEKDAY_COLUMNS, dtype=float)
    table["intervals"] = np.append(n_d, valid.sum())
    with np.errstate(invalid="ignore", divide="ignore"):  # a weekday without intervals gives NaN
        for name, x in (("calendar_days", days), ("pnl", pnl), ("p_step", step), ("residual", resid)):
            table[name] = np.append(by_weekday(x).sum(axis=0) / n_d, x[valid].mean())
    table["share"] = np.append(resid_w.sum(axis=0), resid[valid].sum()) / resid[valid].sum()

    if boot_idx is not None:
        # Weight each window by how often a resample draws it; the weekday means are ratios of
        # weighted sums, so no (B, W, N) array is needed.
        n_resamples, n_windows = boot_idx.shape
        cells = (np.arange(n_resamples)[:, None] * n_windows + boot_idx).ravel()
        weights = np.bincount(cells, minlength=n_resamples * n_windows).reshape(n_resamples, n_windows)
        sums = np.column_stack([resid_w, resid_w.sum(axis=1)])
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (weights @ sums) / (weights @ np.column_stack([counts, counts.sum(axis=1)]))
        tails = [50.0 * (1.0 - level), 100.0 - 50.0 * (1.0 - level)]
        table["residual_lo"], table["residual_hi"] = np.percentile(boot, tails, axis=0)

    grand_mean = resid[valid].mean()
    between = np.sum(n_d[n_d > 0] * (table["residual"].to_numpy()[:-1][n_d > 0] - grand_mean) ** 2)
    eta2 = between / np.sum((resid[valid] - grand_mean) ** 2)
    return table, eta2
