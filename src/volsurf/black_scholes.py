"""Black-Scholes pricing and Greeks for European options.

The core is Black-76 in forward form; the spot functions wrap it with
F = S·exp((r - q)T) and D = exp(-rT).

Conventions
-----------
T      : time to expiry in years (ACT/365 calendar), T > 0
sigma  : annualised volatility as a decimal, sigma > 0
r, q   : continuously compounded rate and dividend yield as decimals
is_call: bool or bool array; True for calls, False for puts

All functions broadcast their inputs with NumPy rules and return arrays of
the broadcast shape (NumPy scalars for scalar inputs).
"""

import numpy as np
from scipy.stats import norm


def _omega(is_call):
    """Sign +1 for calls and -1 for puts, so one formula covers both."""
    return np.where(is_call, 1.0, -1.0)


def _d1_d2(F, K, T, sigma):
    """Black-76 d1 and d2.

    Parameters
    ----------
    F, K : array_like
        Forward and strike (same currency units).
    T : array_like
        Time to expiry in years.
    sigma : array_like
        Volatility as a decimal.

    Returns
    -------
    d1, d2 : ndarray
        d1 = [ln(F/K) + sigma²T/2]/(sigma·√T), d2 = d1 - sigma·√T.
    """
    sig_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / sig_sqrt_t
    return d1, d1 - sig_sqrt_t


def black76_price(F, K, T, D, sigma, is_call=True):
    """Black-76 price of a European option in forward form.

    Parameters
    ----------
    F : array_like
        Forward price for expiry T.
    K : array_like
        Strike.
    T : array_like
        Time to expiry in years.
    D : array_like
        Discount factor to expiry, exp(-rT).
    sigma : array_like
        Volatility as a decimal.
    is_call : bool or array_like of bool
        True for a call, False for a put.

    Returns
    -------
    ndarray
        Call D·[F·N(d1) - K·N(d2)] or put D·[K·N(-d2) - F·N(-d1)], in the
        currency units of F and K.
    """
    F, K, T, D, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, D, sigma))
    w = _omega(is_call)
    d1, d2 = _d1_d2(F, K, T, sigma)
    return D * w * (F * norm.cdf(w * d1) - K * norm.cdf(w * d2))


def black76_vega(F, K, T, D, sigma):
    """Vega of the Black-76 price, dV/dsigma, identical for calls and puts.

    Parameters
    ----------
    F, K, T, D, sigma
        As in black76_price.

    Returns
    -------
    ndarray
        D·F·φ(d1)·√T, in the currency units of F per unit of volatility (a
        move of 1.0, that is 100 vol points). With F = S·exp((r - q)T) and
        D = exp(-rT) it equals bs_vega, since D·F = S·exp(-qT).
    """
    F, K, T, D, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, D, sigma))
    d1, _ = _d1_d2(F, K, T, sigma)
    return D * F * norm.pdf(d1) * np.sqrt(T)


def _spot_inputs(S, K, T, r, q, sigma):
    """Cast spot inputs to float arrays and return them with F and d1, d2."""
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    F = S * np.exp((r - q) * T)
    d1, d2 = _d1_d2(F, K, T, sigma)
    return S, K, T, r, q, sigma, F, d1, d2


def bs_price(S, K, T, r, q, sigma, is_call=True):
    """Black-Scholes price of a European option on a spot with dividend yield.

    Parameters
    ----------
    S : array_like
        Spot price.
    K : array_like
        Strike.
    T : array_like
        Time to expiry in years.
    r : array_like
        Continuously compounded risk-free rate.
    q : array_like
        Continuously compounded dividend yield.
    sigma : array_like
        Volatility as a decimal.
    is_call : bool or array_like of bool
        True for a call, False for a put.

    Returns
    -------
    ndarray
        Option price in the currency units of S, computed as
        black76_price with F = S·exp((r - q)T) and D = exp(-rT).
    """
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    F = S * np.exp((r - q) * T)
    D = np.exp(-r * T)
    return black76_price(F, K, T, D, sigma, is_call)


def bs_delta(S, K, T, r, q, sigma, is_call=True):
    """Spot delta, dV/dS.

    Parameters
    ----------
    S, K, T, r, q, sigma, is_call
        As in bs_price.

    Returns
    -------
    ndarray
        exp(-qT)·N(d1) for a call, -exp(-qT)·N(-d1) for a put (shares of
        spot per option).
    """
    S, K, T, r, q, sigma, F, d1, d2 = _spot_inputs(S, K, T, r, q, sigma)
    w = _omega(is_call)
    return w * np.exp(-q * T) * norm.cdf(w * d1)


def bs_gamma(S, K, T, r, q, sigma):
    """Spot gamma, d²V/dS², identical for calls and puts.

    Parameters
    ----------
    S, K, T, r, q, sigma
        As in bs_price.

    Returns
    -------
    ndarray
        exp(-qT)·φ(d1)/(S·sigma·√T), per unit of spot.
    """
    S, K, T, r, q, sigma, F, d1, d2 = _spot_inputs(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, q, sigma):
    """Vega, dV/dsigma, identical for calls and puts.

    Parameters
    ----------
    S, K, T, r, q, sigma
        As in bs_price.

    Returns
    -------
    ndarray
        S·exp(-qT)·φ(d1)·√T, per unit of volatility (a move of 1.0, that is
        100 vol points; divide by 100 for the change per vol point).
    """
    S, K, T, r, q, sigma, F, d1, d2 = _spot_inputs(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S, K, T, r, q, sigma, is_call=True):
    """Theta, dV/dt as calendar time passes (equal to -dV/dT), per year.

    Parameters
    ----------
    S, K, T, r, q, sigma, is_call
        As in bs_price.

    Returns
    -------
    ndarray
        -S·exp(-qT)·φ(d1)·sigma/(2√T)
        + ω·[q·S·exp(-qT)·N(ω·d1) - r·K·exp(-rT)·N(ω·d2)],
        with ω = +1 for calls and -1 for puts; currency units per year.
    """
    S, K, T, r, q, sigma, F, d1, d2 = _spot_inputs(S, K, T, r, q, sigma)
    w = _omega(is_call)
    disc_spot = S * np.exp(-q * T)
    decay = -disc_spot * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    carry = w * (q * disc_spot * norm.cdf(w * d1) - r * K * np.exp(-r * T) * norm.cdf(w * d2))
    return decay + carry
