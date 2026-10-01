"""Numerical implied volatility extraction from market option prices.

black76_implied_vol is the solver the study uses: Brent's method on a fixed
volatility bracket, after a no-arbitrage check and a sign-change check, with
NaN wherever no implied vol exists. newton_naive and newton_manaster_koehler
are for the demonstration only: Newton's method from a fixed start can leave
the bracket deep out of the money, where vega is close to zero, and the
Manaster-Koehler start fixes that.

Conventions as in black_scholes: F forward, K strike, T years (ACT/365),
D discount factor exp(-rT), volatility as a decimal, is_call True for calls
and False for puts.
"""

import numpy as np
from scipy.optimize import brentq

from volsurf.black_scholes import black76_price, black76_vega

_SIGMA_LO, _SIGMA_HI = 0.001, 5.0  # Brent bracket for the volatility
_XTOL = 1e-14  # Brent tolerance on the volatility


def _no_arbitrage_bounds(F, K, D, is_call):
    """Strict bounds on a European price: D·max(F - K, 0) < C < D·F, D·max(K - F, 0) < P < D·K."""
    lower = D * np.maximum(np.where(is_call, F - K, K - F), 0.0)
    upper = D * np.where(is_call, F, K)
    return lower, upper


def _brent_vol(price, F, K, T, D, is_call):
    """Implied vol of one in-bounds price, or NaN without a sign change on the bracket."""

    def excess(sigma):
        return float(black76_price(F, K, T, D, sigma, is_call)) - price

    if not excess(_SIGMA_LO) < 0.0 < excess(_SIGMA_HI):
        return np.nan
    return brentq(excess, _SIGMA_LO, _SIGMA_HI, xtol=_XTOL)


def black76_implied_vol(price, F, K, T, D, is_call=True):
    """Black-76 implied volatility by Brent's method.

    Each element is solved on its own with scipy.optimize.brentq on the
    bracket [0.001, 5.0] with xtol 1e-14. The Black-76 price rises strictly
    with sigma, so a root exists in the bracket exactly when the price at
    0.001 lies below the target and the price at 5.0 above it; the bracket is
    checked for that sign change first.

    Parameters
    ----------
    price : array_like
        Option price in the currency units of F and K (a bid-ask mid).
    F, K, T, D : array_like
        Forward, strike, time to expiry in years and discount factor.
    is_call : bool or array_like of bool
        True for a call, False for a put.

    Returns
    -------
    ndarray
        Implied volatility as a decimal, of the broadcast shape (a NumPy
        scalar for scalar inputs). NaN where an input is not finite, where
        the price lies outside the strict no-arbitrage bounds
        D·max(F - K, 0) < C < D·F for a call and D·max(K - F, 0) < P < D·K
        for a put, or where the bracket shows no sign change (an implied vol
        below 0.001 or above 5.0).
    """
    price, F, K, T, D, is_call = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, F, K, T, D)), np.asarray(is_call, dtype=bool)
    )
    lower, upper = _no_arbitrage_bounds(F, K, D, is_call)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(price) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T) & np.isfinite(D)
        valid &= (price > lower) & (price < upper)
    price, F, K, T, D, is_call = (x.ravel() for x in (price, F, K, T, D, is_call))
    sigma = np.full(valid.size, np.nan)
    for i in np.flatnonzero(valid):
        sigma[i] = _brent_vol(price[i], F[i], K[i], T[i], D[i], bool(is_call[i]))
    return sigma.reshape(valid.shape)[()]


def _newton(price, F, K, T, D, is_call, sigma0, tol, max_iter):
    """Newton iterates sigma <- sigma - (V(sigma) - price)/vega(sigma) from sigma0.

    Stops as converged when a step moves sigma by less than tol, and as
    failed when an iterate is not positive or not finite, or after max_iter
    steps. Returns (iterates, converged).
    """
    iterates = [float(sigma0)]
    for _ in range(max_iter):
        sigma = iterates[-1]
        if not (np.isfinite(sigma) and sigma > 0):
            return np.array(iterates), False
        step = (float(black76_price(F, K, T, D, sigma, is_call)) - price) / float(black76_vega(F, K, T, D, sigma))
        iterates.append(sigma - step)
        if abs(step) < tol and iterates[-1] > 0:
            return np.array(iterates), True
    return np.array(iterates), False


def newton_naive(price, F, K, T, D, is_call=True, sigma0=0.2, tol=1e-10, max_iter=50):
    """Newton's method for the Black-76 implied vol from a fixed start (demonstration only).

    Deep out of the money and at a low start, vega is close to zero and the
    first step overshoots far beyond the root; the next step can then give a
    negative volatility, where the run fails.

    Parameters
    ----------
    price, F, K, T, D, is_call
        Scalars, as in black76_implied_vol.
    sigma0 : float
        Starting volatility as a decimal.
    tol : float
        Converged when a step moves sigma by less than tol.
    max_iter : int
        Largest number of steps.

    Returns
    -------
    iterates : ndarray
        sigma0 followed by every iterate, the last one included when the run
        stops at a volatility that is not positive.
    converged : bool
        True if a step fell below tol at a positive volatility.
    """
    return _newton(float(price), F, K, T, D, bool(is_call), sigma0, tol, max_iter)


def newton_manaster_koehler(price, F, K, T, D, is_call=True, tol=1e-10, max_iter=50):
    """Newton's method from the Manaster-Koehler start (demonstration only).

    The start sigma0 = √(2|ln(F/K)|/T) is the volatility at which vega peaks
    in sigma, the inflection point of the price in sigma: the price is convex
    below it and concave above it. From there Newton's method converges
    monotonically to the root on either side. At K = F the start is 0, so the
    run fails at once.

    Parameters
    ----------
    price, F, K, T, D, is_call, tol, max_iter
        As in newton_naive.

    Returns
    -------
    iterates : ndarray
        The start followed by every iterate.
    converged : bool
        True if a step fell below tol at a positive volatility.
    """
    sigma0 = np.sqrt(2.0 * abs(np.log(F / K)) / T)
    return _newton(float(price), F, K, T, D, bool(is_call), sigma0, tol, max_iter)
