"""Static arbitrage checks of an SVI surface (DESIGN.md section 8).

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
quoted range and in the extrapolated region separately.

Units: k and w are dimensionless (w is a decimal variance times years), T
is in years and implied vols are decimals.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from volsurf.svi import (
    S_MIN,
    SVIFit,
    _bound_flags,
    _project_inner,
    _raw_from_inner,
    _slice_arrays,
    outer_starts,
    quasi_explicit_inner,
    raw_svi,
    svi_objective,
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
