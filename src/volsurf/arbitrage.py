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

import numpy as np
import pandas as pd

K_RANGE = (-2.0, 1.0)  # log-moneyness range of every check
N_REFIT = 201  # points that constrain the refit
N_FINE = 6001  # fine grid: the exchange rounds of the refit
N_CERTIFY = 60001  # certification grid, never used as a constraint
VIOLATION_TOL = 1e-12  # a margin below -VIOLATION_TOL is a violation


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
