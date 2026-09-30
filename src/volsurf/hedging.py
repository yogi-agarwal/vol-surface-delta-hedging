"""Delta-hedging simulation and P&L attribution (DESIGN.md section 3).

A window is a path of closes S_0, ..., S_n at times t_0 < ... < t_n (years,
ACT/365). At the t_0 close the trader buys one European call struck at K at
its Black-Scholes value with fixed implied vol sigma_i, and holds a short
delta hedge that is reset every h closes and unwound at the t_n close, when
the call pays max(S_n - K, 0). All cash (premium, stock trades, costs) sits in
one account that accrues at the daily rate r_j from close j to close j + 1.

The path is treated as a total-return series: no dividend cash flows are
credited or charged. A dividend yield q, if given, enters pricing and Greeks
only; Stage 5 uses q = 0 because yfinance adjusted closes already contain
dividends.

Every function works on the last axis and broadcasts over leading axes, so a
single call can hedge many paths at once.
"""

from typing import NamedTuple

import numpy as np

from volsurf.black_scholes import bs_delta, bs_gamma, bs_price, bs_vega


class HedgeResult(NamedTuple):
    """Output of simulate_window.

    P&L quantities are fractions of C0 measured at the t_n close (forward
    value), and pnl = payoff + premium + stock + financing + costs exactly.

    Attributes
    ----------
    pnl : ndarray
        Final value of the hedged position.
    payoff : ndarray
        Call payoff max(S_n - K, 0).
    premium : ndarray
        Premium paid at t_0, -C0 (so -1 in units of C0).
    stock : ndarray
        Price P&L of the short stock, -sum_k Delta_k·(S_{k+1} - S_k).
    financing : ndarray
        Interest earned (negative if paid) on the cash account.
    costs : ndarray
        Transaction costs, -c times notional traded, on every trade.
    turnover : ndarray
        Interior turnover sum |Delta_k - Delta_{k-1}| over the rebalances
        strictly between the opening and closing trades, in shares per
        option (not a fraction of C0).
    p_gap : ndarray
        Vol-gap predictor vega_0·(sigma_r² - sigma_i²)/(2·sigma_i).
    p_path : ndarray
        Gamma-path predictor ½·(sigma_r² - sigma_i²)·sum_k Gamma_k·S_k²·dt_k.
    p_step : ndarray
        Step predictor sum_k ½·Gamma_k·S_k²·(R_k² - sigma_i²·dt_k).
    rv : ndarray
        Realised total variance sum_j ln(S_{j+1}/S_j)² over daily closes
        (not annualised).
    c0 : ndarray
        Initial premium C0 in currency units.
    n_intervals : int
        Number of hedge intervals N in the window.
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
    rv: np.ndarray
    c0: np.ndarray
    n_intervals: int


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

    Parameters
    ----------
    n_steps : int
        Number of daily steps n in the window (closes 0, ..., n).
    h : int
        Rebalance interval in closes, h >= 1.

    Returns
    -------
    ndarray of int
        [0, h, 2h, ...] (all below n) followed by n. Its length is N + 1,
        where N = ceil(n/h) is the number of hedge intervals; the last
        interval is shorter than h when h does not divide n.
    """
    if n_steps < 1:
        raise ValueError("a window needs at least one step")
    if h < 1:
        raise ValueError("rebalance interval h must be at least 1")
    return np.append(np.arange(0, n_steps, h), n_steps)


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


def simulate_window(S, t, r, sigma_i, K, h, c, q=0.0):
    """Delta-hedge a long European call along price paths.

    Parameters
    ----------
    S : array_like of shape (..., n + 1)
        Closes S_0, ..., S_n (n >= 1); leading axes index paths.
    t : array_like of shape (..., n + 1)
        Increasing times of the closes in years (ACT/365); only differences
        matter. The call expires at t_n, so T0 = t_n - t_0.
    r : array_like, broadcastable to S
        Continuously compounded rate observed at each close. r_j prices and
        hedges at close j and accrues cash from close j to close j + 1
        (r_n is unused).
    sigma_i : array_like, broadcastable to S.shape[:-1]
        Implied vol as a decimal, fixed for the window.
    K : array_like, broadcastable to S.shape[:-1]
        Strike.
    h : int
        Rebalance interval in closes: the hedge is set at closes 0, h, 2h, ...
        below n and unwound at close n.
    c : float
        Proportional cost as a decimal fraction of notional traded
        (5 bp = 0.0005), charged on every trade including the opening and
        closing ones.
    q : array_like, broadcastable to S.shape[:-1], default 0
        Dividend yield used in pricing and Greeks only (see module docstring).

    Returns
    -------
    HedgeResult
        P&L, its components and the three predictors, each a fraction of C0
        with shape S.shape[:-1], plus turnover, rv, c0 and N. Delta and Gamma
        at close k use sigma_i, rate r_k and remaining time t_n - t_k;
        R_k = S_{k+1}/S_k - 1 and dt_k = t_{k+1} - t_k run over the hedge
        grid, and sigma_r² = rv/T0.
    """
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma_i, K, q = (np.asarray(x, dtype=float) for x in (sigma_i, K, q))
    lead = np.broadcast_shapes(
        S.shape[:-1], t.shape[:-1], r.shape[:-1], sigma_i.shape, K.shape, q.shape
    )
    shape = lead + S.shape[-1:]
    S, t, r = (np.broadcast_to(x, shape) for x in (S, t, r))
    sigma_i, K, q = (np.broadcast_to(x, lead) for x in (sigma_i, K, q))
    if np.any(np.diff(t, axis=-1) <= 0):
        raise ValueError("times must be strictly increasing")

    n = shape[-1] - 1
    grid = hedge_grid(n, h)
    n_intervals = grid.size - 1
    tau = t[..., -1:] - t  # remaining time at each close
    T0 = tau[..., 0]
    sig, strike, div = sigma_i[..., None], K[..., None], q[..., None]

    # Hedge grid: closes where the hedge is set (all but the last grid point).
    S_g, t_g = S[..., grid], t[..., grid]
    S_set, tau_set, r_set = S_g[..., :-1], tau[..., grid[:-1]], r[..., grid[:-1]]
    delta = bs_delta(S_set, strike, tau_set, r_set, div, sig, is_call=True)
    gamma = bs_gamma(S_set, strike, tau_set, r_set, div, sig)
    c0 = bs_price(S[..., 0], K, T0, r[..., 0], q, sigma_i, is_call=True)
    vega0 = bs_vega(S[..., 0], K, T0, r[..., 0], q, sigma_i)

    # Stock position -Delta_k after each grid close, 0 after the unwind.
    position = np.concatenate([-delta, np.zeros(lead + (1,))], axis=-1)
    trade = np.diff(position, axis=-1, prepend=0.0)  # shares bought at each grid close
    cost_flow = -c * np.abs(trade) * S_g
    payoff = np.maximum(S[..., -1] - K, 0.0)
    stock = -np.sum(delta * np.diff(S_g, axis=-1), axis=-1)  # = -sum(trade·S_g)
    costs = np.sum(cost_flow, axis=-1)

    # Growth of one unit of cash from close j to close n.
    log_growth = r[..., :-1] * np.diff(t, axis=-1)
    tail = np.cumsum(log_growth[..., ::-1], axis=-1)[..., ::-1]
    growth = np.exp(np.concatenate([tail, np.zeros(lead + (1,))], axis=-1))
    flow = -trade * S_g + cost_flow  # cash in at each grid close
    financing = np.sum(flow * (growth[..., grid] - 1.0), axis=-1) - c0 * (growth[..., 0] - 1.0)

    pnl = payoff - c0 + stock + financing + costs
    turnover = np.sum(np.abs(trade[..., 1:-1]), axis=-1)

    # Predictors on the hedge grid.
    rv = np.sum(np.diff(np.log(S), axis=-1) ** 2, axis=-1)
    var_gap = rv / T0 - sigma_i**2
    dt_g = np.diff(t_g, axis=-1)
    ret_g = S_g[..., 1:] / S_g[..., :-1] - 1.0
    dollar_gamma = gamma * S_set**2
    p_gap = vega0 * var_gap / (2.0 * sigma_i)
    p_path = 0.5 * var_gap * np.sum(dollar_gamma * dt_g, axis=-1)
    p_step = 0.5 * np.sum(dollar_gamma * (ret_g**2 - sig**2 * dt_g), axis=-1)

    return HedgeResult(
        pnl=pnl / c0,
        payoff=payoff / c0,
        premium=-np.ones(lead),
        stock=stock / c0,
        financing=financing / c0,
        costs=costs / c0,
        turnover=turnover,
        p_gap=p_gap / c0,
        p_path=p_path / c0,
        p_step=p_step / c0,
        rv=rv,
        c0=c0,
        n_intervals=n_intervals,
    )
