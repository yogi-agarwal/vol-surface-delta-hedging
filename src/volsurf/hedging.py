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
(P, n). Windows of different lengths go through simulate_windows.
"""

from typing import NamedTuple

import numpy as np

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
    mu : float
        Drift per year as a decimal (arithmetic, so E[S_t] = S0·exp(mu·t)).
    sigma : float or array_like of shape (n_paths,)
        Volatility per year as a decimal; an array gives each path its own.
    t : array_like of shape (n + 1,)
        Increasing times in years; t[0] is the time of S0.
    n_paths : int
        Number of paths.
    seed : int
        Seed for np.random.default_rng.

    Returns
    -------
    ndarray of shape (n_paths, n + 1)
        Prices, with column 0 equal to S0.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    sigma = np.asarray(sigma, dtype=float)[..., None]
    z = rng.standard_normal((n_paths, dt.size))
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
    """R² of predictions about the 45 degree line.

    Parameters
    ----------
    y, p : array_like of the same shape
        Outcomes and predictions (same units).

    Returns
    -------
    float
        1 - sum (y - p)² / sum (y - mean(y))². Equals 1 for perfect
        predictions and 0 when p is the sample mean; negative when p does
        worse than the mean. Unlike the OLS R², it penalises any intercept
        or slope away from 1.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


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
    S, t, r = (np.asarray(x, dtype=float) for x in (S, t, r))
    if S.ndim != 1 or t.shape != S.shape or r.shape != S.shape:
        raise ValueError("S, t and r must be 1-D histories of the same length")
    starts, ends = np.asarray(starts), np.asarray(ends)
    if starts.ndim != 1 or ends.shape != starts.shape:
        raise ValueError("starts and ends must be 1-D arrays of the same length")
    if not (np.issubdtype(starts.dtype, np.integer) and np.issubdtype(ends.dtype, np.integer)):
        raise ValueError("starts and ends must be integer indices")
    if np.any(starts < 0) or np.any(ends <= starts) or np.any(ends >= S.size):
        raise ValueError("every window needs 0 <= start < end < len(S)")

    n_windows = starts.size
    per_window = []
    for x, name in ((sigma_i, "sigma_i"), (K, "K"), (q, "q")):
        x = np.asarray(x, dtype=float)
        _check_shape(x, name, [(), (n_windows,)])
        per_window.append(np.broadcast_to(x, (n_windows,)))
    sigma_i, K, q = per_window

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
