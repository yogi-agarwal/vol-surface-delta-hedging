"""Static arbitrage checks of an SVI surface, its constrained refit and the variance-swap bridge (DESIGN.md section 8).

Butterfly: a slice w(k) of total implied variance against log-moneyness
k = ln(K/F) is free of butterfly arbitrage when Gatheral and Jacquier's

    g(k) = (1 - k·w'/(2w))² - (w'²/4)·(1/w + 1/4) + w''/2

is non-negative everywhere, because the risk-neutral density of
k = ln(S_T/F) is

    p(k) = g(k)/√(2πw)·exp(-d_-²/2),  d_- = -k/√w - √w/2,

the second derivative of the forward call price in K, times K (Breeden and
Litzenberger). Calendar: total variance may not fall as T rises at fixed k,
w_{i+1}(k) >= w_i(k) for consecutive expiries.

The checks run on grids over k in [-2, 1]: 201 points constrain the refit,
the 6,001-point fine grid feeds its exchange rounds, and 60,001 points
certify the result. A margin below -1e-12 counts as a violation, inside the
quoted range and in the extrapolated region separately. refit_surface refits
the slices in order of expiry under these constraints.

Bridge: the variance-swap total variance of a slice is 2·∫ OTM(K)/K dk
by replication with out-of-the-money options (forward prices per unit of strike),
cross-checked by Gatheral's z-integral over the full range and by the CBOE
discrete formula over quoted strikes. A 30-day slice comes from linear
interpolation of w in T at fixed k between the bracketing expiries.

surface_from_chain runs the whole surface track on one cleaned chain, and
bridge_30d the bridge on its result, for the stability check across
snapshots.

Units: k and w are dimensionless (w is a decimal variance times years), T
is in years and implied vols are decimals.
"""

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import norm

from volsurf.black_scholes import black76_price
from volsurf.implied_vol import black76_implied_vol
from volsurf.svi import (
    S_MIN,
    SVIFit,
    _bound_flags,
    _project_inner,
    _raw_from_inner,
    _slice_arrays,
    fit_svi,
    outer_starts,
    quasi_explicit_inner,
    raw_svi,
    svi_objective,
    vega_weights,
)

K_RANGE = (-2.0, 1.0)  # log-moneyness range of every check
N_REFIT = 201  # points that constrain the refit
N_FINE = 6001  # fine grid: the exchange rounds of the refit
N_CERTIFY = 60001  # certification grid, never used as a constraint
VIOLATION_TOL = 1e-12  # a margin below -VIOLATION_TOL is a violation
_MAX_ROUNDS = 10  # exchange rounds of the refit after its first solve
_MINIMISER_XTOL = 1e-12  # tolerance in k of the bounded search for a margin minimum between fine-grid points
_SLSQP_OPTIONS = {"ftol": 1e-15, "maxiter": 1000}
_SLSQP_RUNS = 3  # SLSQP is restarted from its own point when it reports failure, at most this many runs in all
T_BRIDGE = 30 / 365  # maturity of the variance-swap bridge, years
_QUAD_LIMIT, _QUAD_EPSABS, _QUAD_EPSREL = 1000, 1e-15, 1e-13  # adaptive quadrature of the bridge
_Z_MAX = 12.0  # Gatheral's z-integral runs over [-12, 12]


def k_grid(n):
    """Equally spaced log-moneyness over [-2, 1].

    Parameters
    ----------
    n : int
        Number of points, ends included (201, 6,001 or 60,001).

    Returns
    -------
    ndarray of shape (n,)
    """
    return np.linspace(*K_RANGE, n)


def svi_derivatives(k, a, b, rho, m, s):
    """Raw SVI total variance and its first two derivatives in k.

    With u = k - m and R = √(u² + s²): w = a + b·(rho·u + R),
    w' = b·(rho + u/R) and w'' = b·s²/R³.

    Parameters
    ----------
    k : array_like
        Log-moneyness ln(K/F).
    a, b, rho, m, s : float
        Raw SVI parameters.

    Returns
    -------
    w, dw, d2w : ndarray
        w(k), w'(k) and w''(k), the shape of k.
    """
    u = np.asarray(k, dtype=float) - m
    R = np.sqrt(u * u + s * s)
    return a + b * (rho * u + R), b * (rho + u / R), b * s * s / R**3


def _g(k, w, dw, d2w):
    """Gatheral and Jacquier's g from k, w, w' and w''."""
    first = 1.0 - k * dw / (2.0 * w)
    return first * first - 0.25 * dw * dw * (1.0 / w + 0.25) + 0.5 * d2w


def butterfly_g(k, a, b, rho, m, s):
    """Gatheral and Jacquier's butterfly function g(k) of a raw SVI slice.

    g(k) = (1 - k·w'/(2w))² - (w'²/4)·(1/w + 1/4) + w''/2; the slice is free
    of butterfly arbitrage where g >= 0.

    Parameters
    ----------
    k : array_like
        Log-moneyness ln(K/F).
    a, b, rho, m, s : float
        Raw SVI parameters, with w > 0 at every k.

    Returns
    -------
    ndarray
        g(k), dimensionless, the shape of k.
    """
    k = np.asarray(k, dtype=float)
    return _g(k, *svi_derivatives(k, a, b, rho, m, s))


def svi_density(k, a, b, rho, m, s):
    """Risk-neutral density of k = ln(S_T/F) implied by a raw SVI slice.

    p(k) = g(k)/√(2πw)·exp(-d_-²/2) with d_- = -k/√w - √w/2. It equals
    K·∂²C/∂K² for the forward call price C per unit of F, so it is negative
    exactly where g is.

    Parameters
    ----------
    k : array_like
        Log-moneyness ln(K/F).
    a, b, rho, m, s : float
        Raw SVI parameters, with w > 0 at every k.

    Returns
    -------
    ndarray
        p(k), per unit of k, the shape of k.
    """
    k = np.asarray(k, dtype=float)
    w, dw, d2w = svi_derivatives(k, a, b, rho, m, s)
    root = np.sqrt(w)
    d_minus = -k / root - 0.5 * root
    return _g(k, w, dw, d2w) / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus * d_minus)


def violation_counts(margin, k, k_lo, k_hi, tol=VIOLATION_TOL):
    """Violations of a margin on a grid, inside and outside a quoted range.

    A point violates when its margin is below -tol or is not a number.

    Parameters
    ----------
    margin : array_like
        The margin at each grid point (g, or w_i - w_{i-1}).
    k : array_like
        The grid, the shape of margin.
    k_lo, k_hi : float
        The quoted range; points with k_lo <= k <= k_hi count as quoted.
    tol : float, default 1e-12
        Rounding tolerance.

    Returns
    -------
    quoted, extrapolated : int
        Violations inside and outside [k_lo, k_hi].
    """
    margin, k = np.asarray(margin, dtype=float), np.asarray(k, dtype=float)
    if margin.shape != k.shape:
        raise ValueError("margin and k must have the same shape")
    bad = ~(margin >= -tol)
    inside = (k >= k_lo) & (k <= k_hi)
    return int((bad & inside).sum()), int((bad & ~inside).sum())


def arbitrage_table(params, ranges, k, index=None, tol=VIOLATION_TOL):
    """Butterfly and calendar violations of SVI slices on a grid, one row per slice.

    The butterfly margin of a slice is g(k), quoted inside the slice's own
    quoted range. The calendar margin of slice i is w_i(k) - w_{i-1}(k),
    quoted where k lies inside both slices' quoted ranges; the first slice
    has no calendar margin.

    Parameters
    ----------
    params : sequence of (a, b, rho, m, s)
        Raw SVI parameters of each slice, in increasing order of T.
    ranges : sequence of (k_lo, k_hi)
        Quoted range of each slice, in the same order.
    k : array_like
        The grid (k_grid).
    index : sequence, optional
        Row labels, for example the expiries.
    tol : float, default 1e-12
        Rounding tolerance of violation_counts.

    Returns
    -------
    pd.DataFrame
        Columns "butterfly quoted", "butterfly extrapolated" and "min g", then
        "calendar quoted", "calendar extrapolated" and "min calendar margin"
        (total variance), missing on the first row. Counts are integers.
    """
    if len(params) != len(ranges):
        raise ValueError("params and ranges must have one entry per slice")
    k = np.asarray(k, dtype=float)
    rows, previous = [], None
    for p, (k_lo, k_hi) in zip(params, ranges):
        g = butterfly_g(k, *p[:5])
        row = dict(zip(["butterfly quoted", "butterfly extrapolated"], violation_counts(g, k, k_lo, k_hi, tol)))
        row["min g"] = float(g.min())
        w = svi_derivatives(k, *p[:5])[0]
        if previous is not None:
            w_prev, (lo_prev, hi_prev) = previous
            margin = w - w_prev
            counts = violation_counts(margin, k, max(k_lo, lo_prev), min(k_hi, hi_prev), tol)
            row.update(zip(["calendar quoted", "calendar extrapolated"], counts))
            row["min calendar margin"] = float(margin.min())
        rows.append(row)
        previous = w, (k_lo, k_hi)
    table = pd.DataFrame(rows, index=index, columns=[
        "butterfly quoted", "butterfly extrapolated", "min g",
        "calendar quoted", "calendar extrapolated", "min calendar margin",
    ])
    for column in ("butterfly quoted", "butterfly extrapolated", "calendar quoted", "calendar extrapolated"):
        table[column] = table[column].astype("Int64")
    return table


class RefitResult(NamedTuple):
    """A slice after the constrained refit.

    fit : SVIFit
        The slice: the start itself when it already met every constraint,
        else the constrained minimiser.
    refitted : bool
        False when the start was kept as it was.
    rounds : int
        Exchange rounds after the first solve.
    added : ndarray
        The points added to the 201 constraint points by the exchange
        rounds: fine-grid points and minimisers of a margin between them.
    """

    fit: SVIFit
    refitted: bool
    rounds: int
    added: np.ndarray


def _inner_parts(k, p):
    """w, w', w'' at k and their gradients in p = (a, c, d, m, s), with w = a + d·y + c·√(y² + 1), y = (k - m)/s.

    The gradients have shape (k.size, 5).
    """
    a, c, d, m, s = p
    y = (k - m) / s
    q = np.sqrt(y * y + 1.0)
    w = a + d * y + c * q
    slope = d + c * y / q  # dw/dy
    w1 = slope / s
    w2 = c / (s * s * q**3)
    dy_dm, dy_ds = -1.0 / s, -y / s
    one, zero = np.ones_like(k), np.zeros_like(k)
    dw = np.stack([one, q, y, slope * dy_dm, slope * dy_ds], axis=-1)
    curve = c / q**3 / s  # d(w')/dy
    dw1 = np.stack([zero, y / (q * s), one / s, curve * dy_dm, curve * dy_ds - w1 / s], axis=-1)
    dw2_dy = -3.0 * c * y / (s * s * q**5)
    dw2 = np.stack([zero, 1.0 / (s * s * q**3), zero, dw2_dy * dy_dm, dw2_dy * dy_ds - 2.0 * w2 / s], axis=-1)
    return w, w1, w2, dw, dw1, dw2


def _g_and_jacobian(k, p):
    """g at k and its gradient in p = (a, c, d, m, s), shape (k.size, 5)."""
    w, w1, w2, dw, dw1, dw2 = _inner_parts(k, p)
    first = 1.0 - k * w1 / (2.0 * w)
    g = first * first - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
    g_w = first * k * w1 / (w * w) + w1 * w1 / (4.0 * w * w)
    g_w1 = -first * k / w - 0.5 * w1 * (1.0 / w + 0.25)
    return g, g_w[:, None] * dw + g_w1[:, None] * dw1 + 0.5 * dw2


def _violations(k, params, previous, tol):
    """True at the points of k where g, or the calendar margin against the previous slice, is below -tol."""
    bad = ~(butterfly_g(k, *params) >= -tol)
    if previous is not None:
        bad |= ~(raw_svi(k, *params) - raw_svi(k, *previous) >= -tol)
    return bad


def _margin_minimisers(fine, params, previous, tol):
    """Minimisers of g and of the calendar margin between fine-grid neighbours, where the margin there is below -tol.

    At every discrete local minimum of a margin on the fine grid (ends
    included), a bounded scalar search over the neighbouring interval finds
    the margin's minimum between the grid points, which a grid alone misses.
    """
    margins = [lambda x: butterfly_g(x, *params)]
    if previous is not None:
        margins.append(lambda x: raw_svi(x, *params) - raw_svi(x, *previous))
    found = []
    for margin in margins:
        values = margin(fine)
        lower = np.r_[np.inf, values[:-1]]  # left neighbours, none at the first point
        upper = np.r_[values[1:], np.inf]
        for j in np.flatnonzero((values <= lower) & (values <= upper)):
            lo, hi = fine[max(j - 1, 0)], fine[min(j + 1, fine.size - 1)]
            result = minimize_scalar(lambda x: float(margin(x)), bounds=(lo, hi), method="bounded",
                                     options={"xatol": _MINIMISER_XTOL})
            if not result.fun >= -tol:
                found.append(float(result.x))
    return np.array(found)


def _exchange_points(fine, params, previous, tol):
    """Points to add to the constraints: fine-grid violations and margin minimisers below -tol, sorted."""
    return np.union1d(fine[_violations(fine, params, previous, tol)], _margin_minimisers(fine, params, previous, tol))


def _constrained_solve(k, w, weights, x0, points, previous, k_lo, k_hi):
    """SLSQP solution (a, c, d, m, s) of the refit with constraints at the points (see refit_slice)."""
    v = weights @ w  # (a, c, d) enter the solver divided by v, so its variables are of order 1
    scale = weights @ (w * w)
    sv = np.array([v, v, v, 1.0, 1.0])

    def objective(x):
        a, c, d, m, s = x * sv
        y = (k - m) / s
        q = np.sqrt(y * y + 1.0)
        r = a + d * y + c * q - w
        slope = d + c * y / q
        jac = np.stack([np.ones_like(k), q, y, -slope / s, -slope * y / s], axis=-1) * sv
        return weights @ (r * r) / scale, 2.0 * jac.T @ (weights * r) / scale

    def linear(row):
        row = np.asarray(row, dtype=float)
        return {"type": "ineq", "fun": lambda x: row @ x, "jac": lambda x: row}

    def cone(x):
        return x[1] - np.hypot(x[2], min(x[0], 0.0))

    def cone_jac(x):
        norm = np.hypot(x[2], min(x[0], 0.0))
        if norm == 0.0:
            return np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        return np.array([-min(x[0], 0.0) / norm, 1.0, -x[2] / norm, 0.0, 0.0])

    constraints = [
        linear([0.0, 1.0, -1.0, 0.0, 0.0]),  # c >= d
        linear([0.0, 1.0, 1.0, 0.0, 0.0]),  # c >= -d
        linear([0.0, -v, -v, 0.0, 2.0]),  # c + d <= 2s (Lee)
        linear([0.0, -v, v, 0.0, 2.0]),  # c - d <= 2s (Lee)
        {"type": "ineq", "fun": cone, "jac": cone_jac},  # w >= 0
        {"type": "ineq", "fun": lambda x: _g_and_jacobian(points, x * sv)[0],
         "jac": lambda x: _g_and_jacobian(points, x * sv)[1] * sv},
    ]
    if previous is not None:
        w_prev = raw_svi(points, *previous)
        constraints.append({
            "type": "ineq",
            "fun": lambda x: (_inner_parts(points, x * sv)[0] - w_prev) / v,
            "jac": lambda x: _inner_parts(points, x * sv)[3] * sv / v,
        })
    bounds = [(None, None), (0.0, None), (None, None), (k_lo, k_hi), (S_MIN, None)]
    x = np.asarray(x0, dtype=float) / sv
    for _ in range(_SLSQP_RUNS):
        result = minimize(objective, x, jac=True, method="SLSQP", bounds=bounds, constraints=constraints,
                          options=_SLSQP_OPTIONS)
        x = result.x
        if result.success:
            break
    else:
        raise RuntimeError(f"the constrained refit did not converge: {result.message}")
    a, c, d, m, s = x * sv
    m, s = min(max(m, k_lo), k_hi), max(s, S_MIN)
    return (*_project_inner(a, c, d, s), m, s)


def _raw(p):
    """Raw (a, b, rho, m, s) from (a, c, d, m, s)."""
    a, c, d, m, s = p
    return (*_raw_from_inner(a, c, d, s), m, s)


def refit_slice(k, w, weights, start, previous=None, force=False, tol=VIOLATION_TOL, max_rounds=_MAX_ROUNDS):
    """Refit one SVI slice free of butterfly and calendar arbitrage on the grid (DESIGN.md section 8).

    The objective is the Stage 3 one, f = Σ ω·(w(k) - w)² with the weights
    normalised to sum to 1, and the Stage 3 constraints stay: b >= 0,
    |rho| <= 1, s >= 1e-4, w >= 0, Lee's bound and m in the quoted range.
    The new constraints are g(k_j) >= 0 and w(k_j) >= w_prev(k_j) at the 201
    points of k_grid(201), with w_prev the previous slice.

    When the start meets every new constraint on the 201 points, on the
    6,001-point fine grid and between the fine-grid points (to the
    tolerance), it is returned as it is: as the Stage 3 minimiser it is also
    the constrained one. Otherwise SLSQP
    minimises f/Σ ω·w² over (a, c, d, m, s), with c = b·s, d = rho·b·s and
    (a, c, d) divided by v = Σ ω·w, from the start, with analytic gradients.
    The Stage 3 constraints are c >= |d|, c + |d| <= 2s and the cone
    c >= √(d² + min(a, 0)²), as in svi.quasi_explicit_inner, with bounds on
    m and s. A run that reports failure restarts from its own point, at most
    three runs, and the solution is moved onto the Stage 3 constraint set.
    Then the exchange rounds: every fine-grid point where a margin (g or the
    calendar margin) is below -tol joins the constraint points, and so does
    the minimiser of a margin between fine-grid neighbours when the margin
    there is below -tol. These minimisers come from a bounded scalar search
    over the two intervals next to each discrete local minimum of the margin
    on the fine grid, so a margin that touches zero at grid points and dips
    between them is caught. The problem is solved again from the last
    solution until neither kind of point is left.

    Parameters
    ----------
    k, w : array_like
        Log-moneyness and market total variance of the quotes (1-D).
    weights : array_like
        Non-negative weights (svi.vega_weights); normalised here.
    start : SVIFit or sequence
        The start, raw (a, b, rho, m, s) first: the Stage 3 fit.
    previous : sequence, optional
        Raw (a, b, rho, m, s) of the previous (shorter) slice after its
        refit; None for the first slice, which has no calendar constraint.
    force : bool, default False
        Solve even when the start meets every constraint (the restarts of
        refit_restarts).
    tol : float, default 1e-12
        Violation tolerance.
    max_rounds : int, default 10
        Most exchange rounds after the first solve.

    Returns
    -------
    RefitResult

    Raises
    ------
    RuntimeError
        If SLSQP does not converge in three runs, or a violation on the
        fine grid or between its points is left after max_rounds exchange
        rounds.
    """
    k, w, weights = _slice_arrays(k, w, weights)
    k_lo, k_hi = float(k.min()), float(k.max())
    refit_points, fine = k_grid(N_REFIT), k_grid(N_FINE)
    a, b, rho, m, s = (float(x) for x in start[:5])
    if not force and not (_violations(refit_points, (a, b, rho, m, s), previous, tol).any()
                          or _exchange_points(fine, (a, b, rho, m, s), previous, tol).size):
        fit = start if isinstance(start, SVIFit) else SVIFit(
            a, b, rho, m, s, svi_objective(k, w, weights, a, b, rho, m, s), *_bound_flags(m, rho, k_lo, k_hi))
        return RefitResult(fit, False, 0, np.array([]))

    points = refit_points
    p = _constrained_solve(k, w, weights, (a, b * s, rho * b * s, m, s), points, previous, k_lo, k_hi)
    for rounds in range(max_rounds + 1):
        new = _exchange_points(fine, _raw(p), previous, tol)
        if not new.size:
            break
        if rounds == max_rounds:
            raise RuntimeError(f"{new.size} violations left on the fine grid or between its points "
                               f"after {max_rounds} exchange rounds")
        points = np.union1d(points, new)
        p = _constrained_solve(k, w, weights, p, points, previous, k_lo, k_hi)
    a, b, rho, m, s = _raw(p)
    fit = SVIFit(a, b, rho, m, s, svi_objective(k, w, weights, a, b, rho, m, s), *_bound_flags(m, rho, k_lo, k_hi))
    return RefitResult(fit, True, rounds, np.setdiff1d(points, refit_points))


def refit_surface(slices, starts, tol=VIOLATION_TOL):
    """Refit SVI slices in order of expiry, shortest first, each against the previous refit.

    Parameters
    ----------
    slices : sequence of (k, w, weights)
        Quotes of each slice, in increasing order of T.
    starts : sequence
        The Stage 3 fit of each slice (SVIFit), in the same order.
    tol : float, default 1e-12
        Violation tolerance of refit_slice.

    Returns
    -------
    list of RefitResult
        One per slice, in the given order.
    """
    if len(slices) != len(starts):
        raise ValueError("slices and starts must have one entry per slice")
    results, previous = [], None
    for (k, w, weights), start in zip(slices, starts):
        result = refit_slice(k, w, weights, start, previous, tol=tol)
        results.append(result)
        previous = result.fit[:5]
    return results


def refit_restarts(k, w, weights, previous=None, tol=VIOLATION_TOL):
    """The constrained refit of one slice repeated from the nine Stage 3 outer starts (the cross-check).

    Each start is (m0, s0) from svi.outer_starts with its inner (a, c, d)
    from svi.quasi_explicit_inner, solved with refit_slice(force=True).

    Parameters
    ----------
    k, w, weights, previous, tol
        As in refit_slice.

    Returns
    -------
    list of RefitResult or None
        One per start, None where the solve raised.
    """
    results = []
    for m0, s0 in outer_starts(k, w, weights):
        a, c, d, _ = quasi_explicit_inner(k, w, weights, m0, s0)
        try:
            results.append(refit_slice(k, w, weights, _raw((a, c, d, m0, s0)), previous, force=True, tol=tol))
        except RuntimeError:
            results.append(None)
    return results


def bracket(T, T_slices):
    """The consecutive slices that bracket a maturity, and the weight of the later one.

    Parameters
    ----------
    T : float
        Maturity in years, with T_slices[0] <= T <= T_slices[-1].
    T_slices : array_like
        Maturities of the slices in years, strictly increasing.

    Returns
    -------
    i, j : int
        Indices with j = i + 1 and T_slices[i] <= T <= T_slices[j].
    lam : float
        (T - T_i)/(T_j - T_i), so w(T) = (1 - lam)·w_i + lam·w_j.

    Raises
    ------
    ValueError
        If T_slices is not strictly increasing with at least two entries, or
        T lies outside [T_slices[0], T_slices[-1]].
    """
    T_slices = np.asarray(T_slices, dtype=float)
    if T_slices.ndim != 1 or T_slices.size < 2 or not (np.diff(T_slices) > 0).all():
        raise ValueError("T_slices must be strictly increasing, with at least two slices")
    if not T_slices[0] <= T <= T_slices[-1]:
        raise ValueError(f"T = {T} lies outside the slices, [{T_slices[0]}, {T_slices[-1]}]")
    i = min(int(np.searchsorted(T_slices, T, side="right")) - 1, T_slices.size - 2)
    return i, i + 1, float((T - T_slices[i]) / (T_slices[i + 1] - T_slices[i]))


def interpolate_w(k, T, T_slices, params):
    """Total variance between SVI slices, linear in T at fixed k (DESIGN.md section 8).

    w(k, T) = (1 - lam)·w_i(k) + lam·w_{i+1}(k) with lam = (T - T_i)/(T_{i+1} - T_i)
    for the bracketing slices (bracket). Interpolating total variance at
    fixed k keeps the calendar order of the slices: where w_{i+1} >= w_i,
    w rises with T between them.

    Parameters
    ----------
    k : array_like
        Log-moneyness.
    T : float or array_like
        Maturities in years inside [T_slices[0], T_slices[-1]]; broadcast
        against k.
    T_slices : array_like
        Maturities of the slices in years, strictly increasing.
    params : sequence of (a, b, rho, m, s)
        Raw SVI parameters of each slice, in the order of T_slices.

    Returns
    -------
    ndarray
        w(k, T), the broadcast shape of k and T.

    Raises
    ------
    ValueError
        As for bracket, or if params and T_slices differ in length.
    """
    T_slices = np.asarray(T_slices, dtype=float)
    if len(params) != T_slices.size:
        raise ValueError("params and T_slices must have one entry per slice")
    k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
    if T.size:
        bracket(float(T.min()), T_slices)
        bracket(float(T.max()), T_slices)
    i = np.minimum(np.searchsorted(T_slices, T, side="right") - 1, T_slices.size - 2)
    lam = (T - T_slices[i]) / (T_slices[i + 1] - T_slices[i])
    slices = np.stack([raw_svi(k, *p[:5]) for p in params])  # (n_slices, *shape)
    w_i = np.take_along_axis(slices, i[None], axis=0)[0]
    w_j = np.take_along_axis(slices, (i + 1)[None], axis=0)[0]
    return (1.0 - lam) * w_i + lam * w_j


def otm_per_strike(k, w):
    """Out-of-the-money Black-76 forward price per unit of strike, OTM(K)/K with F = 1.

    The put for k < 0 and the call for k >= 0, undiscounted, at strike
    K = F·e^k and total variance w, divided by K: N(-d_-) - e^(-k)·N(-d_+)
    for the put and e^(-k)·N(d_+) - N(d_-) for the call, with
    d_± = -k/√w ± √w/2. The terms with e^(-k) are taken through
    log N, so the deep wings neither overflow nor lose their scale. The
    replication integrand of variance_swap_replication is twice this.

    Parameters
    ----------
    k : array_like
        Log-moneyness ln(K/F).
    w : array_like
        Total implied variance at k, > 0.

    Returns
    -------
    ndarray
        OTM(K)/K, dimensionless, the broadcast shape of k and w.
    """
    k, w = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(w, dtype=float))
    root = np.sqrt(w)
    d_plus = -k / root + 0.5 * root
    d_minus = d_plus - root
    out = np.empty(k.shape)
    put = k < 0  # each side only where it applies, so e^(-k) never meets the other wing
    out[put] = norm.cdf(-d_minus[put]) - np.exp(-k[put] + norm.logcdf(-d_plus[put]))
    out[~put] = np.exp(-k[~put] + norm.logcdf(d_plus[~put])) - norm.cdf(d_minus[~put])
    return out[()]


def variance_swap_replication(w, k_lo=-np.inf, k_hi=np.inf):
    """Variance-swap total variance by replication with out-of-the-money options, in forward terms.

    σ²_VS·T = 2·∫ OTM(K)/K² dK with forward prices. With K = F·e^k this is
    2·∫ (OTM(K)/K) dk over k = ln(K/F) (otm_per_strike). The integral runs
    over [k_lo, k_hi] by adaptive quadrature (scipy.integrate.quad), split at
    k = 0, where the integrand has a kink. The full range gives the model's
    variance swap; a finite range truncates it, as the VIX does at the last
    quoted strikes.

    Parameters
    ----------
    w : callable
        Total implied variance as a function of k (vectorised or scalar).
    k_lo, k_hi : float
        Integration range in k, k_lo < k_hi; infinite by default.

    Returns
    -------
    float
        σ²_VS·T (decimal variance times years); the vol is √(σ²_VS·T / T).
    """
    if not k_lo < k_hi:
        raise ValueError("the range needs k_lo < k_hi")

    def integrand(k):
        return 2.0 * float(otm_per_strike(k, w(k)))

    pieces = [(k_lo, min(k_hi, 0.0)), (max(k_lo, 0.0), k_hi)]
    return float(sum(quad(integrand, lo, hi, limit=_QUAD_LIMIT, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL)[0]
                     for lo, hi in pieces if lo < hi))


def _d_minus(k, w):
    """d_-(k) = -k/√w - √w/2 of a slice w(k)."""
    root = np.sqrt(w(k))
    return -k / root - 0.5 * root


def variance_swap_z_integral(w):
    """Full-range variance-swap total variance by Gatheral's z-integral.

    σ²_VS·T = ∫ φ(z)·w(k(z)) dz, with φ the standard normal density and k(z)
    the inverse of z = d_-(k) = -k/√w - √w/2, which falls from +∞ to -∞ as k
    rises on a slice free of butterfly arbitrage. z runs over [-12, 12],
    where φ leaves out less than 1e-32; k(z) comes from brentq. The map is
    checked to fall on 20,001 points between k(12) and k(-12).

    Parameters
    ----------
    w : callable
        Total implied variance as a function of k (vectorised).

    Returns
    -------
    float
        σ²_VS·T.

    Raises
    ------
    ValueError
        If d_- does not fall in k over the range.
    """

    def k_of(z):
        lo, hi = -1.0, 1.0
        while _d_minus(lo, w) < z:
            lo *= 2.0
        while _d_minus(hi, w) > z:
            hi *= 2.0
        return brentq(lambda k: _d_minus(k, w) - z, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps)

    k_hi, k_lo = k_of(-_Z_MAX), k_of(_Z_MAX)
    if not (np.diff(_d_minus(np.linspace(k_lo, k_hi, 20001), w)) < 0).all():
        raise ValueError("d_-(k) must fall as k rises; the slice is not free of butterfly arbitrage")
    value, _ = quad(lambda z: math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) * float(w(k_of(z))),
                    -_Z_MAX, _Z_MAX, points=[0.0], limit=_QUAD_LIMIT, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL)
    return float(value)


def vix_discrete_variance(k_nodes, w):
    """Variance-swap total variance by the CBOE discrete formula, in forward terms.

    σ²·T = 2·Σ Δx_j/x_j²·Q(x_j) - (1/x_0 - 1)², with x = K/F = e^k at the
    nodes in increasing order, Q the out-of-the-money forward price per unit
    of F at w (puts below x_0, calls above, and the mean of the put and the
    call at x_0), x_0 the largest node at or below 1, and Δx_j half the
    distance between the neighbours of node j (the distance to the one
    neighbour at the ends). The last term corrects for splitting puts from
    calls at x_0 rather than at the forward.

    Parameters
    ----------
    k_nodes : array_like
        Log-moneyness of the strikes, at least two distinct, one at or
        below 0; duplicates are dropped.
    w : callable
        Total implied variance as a function of k (vectorised).

    Returns
    -------
    float
        σ²·T over the nodes.

    Raises
    ------
    ValueError
        With fewer than two distinct nodes or none at or below k = 0.
    """
    x = np.unique(np.exp(np.asarray(k_nodes, dtype=float)))
    if x.size < 2 or x[0] > 1.0:
        raise ValueError("the formula needs two distinct nodes, one of them at or below the forward")
    k = np.log(x)
    root = np.sqrt(w(k))
    put = black76_price(1.0, x, 1.0, 1.0, root, is_call=False)
    call = black76_price(1.0, x, 1.0, 1.0, root, is_call=True)
    i0 = int(np.flatnonzero(x <= 1.0)[-1])
    q = np.where(np.arange(x.size) < i0, put, call)
    q[i0] = 0.5 * (put[i0] + call[i0])
    dx = np.empty_like(x)
    dx[0], dx[-1] = x[1] - x[0], x[-1] - x[-2]
    dx[1:-1] = 0.5 * (x[2:] - x[:-2])
    return float(2.0 * np.sum(dx / x**2 * q) - (1.0 / x[i0] - 1.0) ** 2)


def vix_cell_range(k_nodes):
    """The range of k that the cells of vix_discrete_variance cover.

    Each node's weight Δx is the width of its cell. The end nodes get the
    full distance to their one neighbour, so with nodes x_1 < ... < x_n
    (x = e^k) the cells run from x_1 - (x_2 - x_1)/2 to x_n + (x_n - x_{n-1})/2,
    half a strike step beyond each end strike. The replication over this
    range is the like-for-like cross-check of the discrete formula.

    Parameters
    ----------
    k_nodes : array_like
        Log-moneyness of the strikes, at least two distinct; duplicates are
        dropped.

    Returns
    -------
    k_lo, k_hi : float
        The ends of the cells in log-moneyness.

    Raises
    ------
    ValueError
        With fewer than two distinct nodes.
    """
    x = np.unique(np.exp(np.asarray(k_nodes, dtype=float)))
    if x.size < 2:
        raise ValueError("the cells need two distinct nodes")
    return math.log(x[0] - 0.5 * (x[1] - x[0])), math.log(x[-1] + 0.5 * (x[-1] - x[-2]))


class Surface(NamedTuple):
    """The Stage 2c to 4 surface of one cleaned chain (surface_from_chain).

    expiries : list of pd.Timestamp
        The expiries, in increasing order of T.
    T : ndarray
        Time to expiry of each slice in years.
    quotes : list of (k, w, weights)
        Each slice's kept quotes with an implied vol, sorted by k: log-moneyness,
        market total variance and Black-76 vega weights summing to 1.
    ranges : list of (k_lo, k_hi)
        Each slice's quoted range of k.
    fits : list of SVIFit
        The Stage 3 fits.
    refits : list of RefitResult
        The Stage 4 constrained refit, slice by slice.
    """

    expiries: list
    T: np.ndarray
    quotes: list
    ranges: list
    fits: list
    refits: list


class Bridge(NamedTuple):
    """The 30-day variance-swap bridge of a surface (bridge_30d); vols as decimals.

    i, j, lam : the bracketing slices and the weight of the later one (bracket).
    k_lo, k_hi : the integration range, inside both slices' quoted ranges.
    n_strikes : distinct quoted strikes of both slices inside the range.
    sigma_vs : variance-swap vol by replication over the range.
    sigma_vs_discrete : the CBOE discrete formula on the same slice at those strikes.
    sigma_vs_cells : replication over the range the formula's cells cover (vix_cell_range),
        its cross-check.
    sigma_vs_cboe : the CBOE order, the formula on each slice's own quoted strikes in
        the range, then σ²T linear in T (information only).
    sigma_atm : ATM vol √(w(0)/T) of the slice.
    gap : sigma_vs - sigma_atm.
    sigma_vs_full, sigma_vs_full_z : the full-range variance-swap vol by replication
        and by Gatheral's z-integral (the sensitivity).
    """

    i: int
    j: int
    lam: float
    k_lo: float
    k_hi: float
    n_strikes: int
    sigma_vs: float
    sigma_vs_discrete: float
    sigma_vs_cells: float
    sigma_vs_cboe: float
    sigma_atm: float
    gap: float
    sigma_vs_full: float
    sigma_vs_full_z: float


def surface_from_chain(chain):
    """Stages 2c to 4 on one cleaned chain: implied vols, Stage 3 SVI fits and the constrained refit.

    The same steps as the notebook's surface sections. Every contract with
    status kept is solved for its Black-76 implied vol at the mid with its
    expiry's F, D and T, and those without one are left out (the no-IV
    filter). Each expiry's remaining quotes, sorted by k, give w = σ²T and
    vega weights; fit_svi fits them, and refit_surface refits the slices in
    order of T.

    Parameters
    ----------
    chain : pd.DataFrame
        A cleaned chain (data.clean_chain or data.load_chain).

    Returns
    -------
    Surface
    """
    kept = chain[chain["status"] == "kept"]
    kept = kept.assign(iv=black76_implied_vol(
        kept["mid"], kept["F"], kept["strike"], kept["T"], kept["D"], kept["option_type"] == "call"
    )).dropna(subset=["iv"])
    expiries, T, quotes, fits = [], [], [], []
    for expiry, rows in kept.groupby("expiry"):
        rows = rows.sort_values("k")
        k, w = rows["k"].to_numpy(), (rows["iv"] ** 2 * rows["T"]).to_numpy()
        weights = vega_weights(rows["F"], rows["strike"], rows["T"], rows["D"], rows["iv"])
        expiries.append(expiry)
        T.append(float(rows["T"].iloc[0]))
        quotes.append((k, w, weights))
        fits.append(fit_svi(k, w, weights))
    ranges = [(float(k.min()), float(k.max())) for k, _, _ in quotes]
    return Surface(expiries, np.array(T), quotes, ranges, fits, refit_surface(quotes, fits))


def bridge_30d(surface, T=T_BRIDGE):
    """The variance-swap bridge of a surface at T (DESIGN.md section 8), from its refitted slices.

    The slice at T is linear in T at fixed k between the bracketing refits
    (interpolate_w), over the intersection of their quoted ranges. The
    variance-swap vol comes by replication over that range. The CBOE discrete
    formula on the same slice at the quoted strikes of both expiries is
    cross-checked against the replication over the range its cells cover
    (vix_cell_range), and the CBOE order is computed for information. The full
    range by replication and by the z-integral is the sensitivity.

    Parameters
    ----------
    surface : Surface
        From surface_from_chain.
    T : float, default 30/365
        Maturity of the bridge in years.

    Returns
    -------
    Bridge
    """
    params = [result.fit[:5] for result in surface.refits]
    i, j, lam = bracket(T, surface.T)
    k_lo, k_hi = max(surface.ranges[i][0], surface.ranges[j][0]), min(surface.ranges[i][1], surface.ranges[j][1])

    def w_at(k):
        return interpolate_w(k, T, surface.T, params)

    def in_range(k):
        return k[(k >= k_lo) & (k <= k_hi)]

    nodes = np.unique(np.concatenate([in_range(surface.quotes[n][0]) for n in (i, j)]))
    per_expiry = [vix_discrete_variance(in_range(surface.quotes[n][0]), lambda k, p=params[n]: raw_svi(k, *p))
                  for n in (i, j)]
    sigma_vs = math.sqrt(variance_swap_replication(w_at, k_lo, k_hi) / T)
    sigma_atm = math.sqrt(float(w_at(0.0)) / T)
    return Bridge(
        i, j, lam, k_lo, k_hi, int(nodes.size),
        sigma_vs,
        math.sqrt(vix_discrete_variance(nodes, w_at) / T),
        math.sqrt(variance_swap_replication(w_at, *vix_cell_range(nodes)) / T),
        math.sqrt(((1.0 - lam) * per_expiry[0] + lam * per_expiry[1]) / T),
        sigma_atm,
        sigma_vs - sigma_atm,
        math.sqrt(variance_swap_replication(w_at) / T),
        math.sqrt(variance_swap_z_integral(w_at) / T),
    )
