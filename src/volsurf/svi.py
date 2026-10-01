"""Raw SVI smiles and their calibration (DESIGN.md section 7).

Raw SVI gives the total implied variance w = sigma²·T of one expiry as a
function of log-moneyness k = ln(K/F):

    w(k) = a + b·(rho·(k - m) + √((k - m)² + s²)),

with level a, slope b, rotation rho, translation m and curvature s (named s,
never sigma). The constraints are b >= 0, |rho| <= 1, s > 0,
a + b·s·√(1 - rho²) >= 0 (the minimum of w, so w >= 0) and
b·(1 + |rho|) <= 2 (Lee's moment bound: neither wing of w grows faster than
2·|k|). rho = ±1 is the limit in which one wing is flat; w stays >= 0 and
Lee's bound still holds there, and fits that reach it are flagged.

A slice is fitted to one expiry's quotes by minimising the vega-weighted
squared error in total variance, f = Σ ω_i·(w(k_i) - w_i)², with the weights
ω normalised to sum to 1.

fit_svi is the quasi-explicit calibration of Zeliade (De Marco and Martini,
2009). For fixed (m, s), y = (k - m)/s turns the smile into
w = a + d·y + c·√(y² + 1), with c = b·s and d = rho·b·s, which is linear in
(a, c, d). The inner problem is weighted least squares in (a, c, d) subject
to c >= 0, |d| <= c, c + |d| <= 2s and a + √(c² - d²) >= 0, a convex
problem; the outer problem searches (m, s) by Nelder-Mead from a grid of
starts, with m inside the quoted range of k. fit_svi_direct fits the five raw
parameters at once by L-BFGS-B from random starts; it is the cross-check.

Units: k and w are dimensionless (w is a decimal variance times years), T is
in years and implied vols are decimals; fit errors are in vol points
(1 vol point = 0.01).
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from volsurf.black_scholes import black76_vega

S_MIN = 1e-4  # lower bound on the curvature s
M_BOUND_TOL = 1e-8  # m within this of k_min or k_max is on its bound
RHO_BOUND_TOL = 1e-8  # |rho| within this of 1 is on its bound
SEED_DIRECT = 20261002  # seed of the random starts of fit_svi_direct
N_DIRECT_STARTS = 20
_DIRECT_S_RANGE = (1e-3, 2.0)  # random starts of s are log-uniform on this range
_START_M = (-1.0, 0.0, 1.0)  # outer starts: m0 in these multiples of the scale √(Σ ω w), clipped into the range
_START_S = (0.5, 1.0, 2.0)  # outer starts: s0 in these multiples of the scale
_NM_OPTIONS = {"xatol": 1e-10, "fatol": 1e-16, "maxiter": 4000, "maxfev": 8000}
_SLSQP_OPTIONS = {"ftol": 1e-15, "maxiter": 500}
_SLSQP_RUNS = 3  # SLSQP is restarted from its own point when it reports failure, at most this many runs in all
_LBFGSB_OPTIONS = {"ftol": 1e-16, "gtol": 1e-14, "maxiter": 20000, "maxfun": 50000}
_ROOT_FLOOR = 1e-12  # floor on √(c² - d²) and √(1 - rho²) in gradients, where they would divide by zero


class SVIFit(NamedTuple):
    """A raw SVI slice fitted to one expiry.

    a, b, rho, m, s : the raw SVI parameters.
    objective : Σ ω·(w(k) - w)² at the fit, with the weights normalised to
        sum to 1 (total variance squared).
    m_on_bound : True when m lies within 1e-8 of the smallest or largest
        quoted k, the bounds of the search.
    rho_on_bound : True when |rho| lies within 1e-8 of 1, where one wing
        of the smile is flat.
    """

    a: float
    b: float
    rho: float
    m: float
    s: float
    objective: float
    m_on_bound: bool
    rho_on_bound: bool


class SVIErrors(NamedTuple):
    """Fit errors of an SVI slice in implied vol, in vol points.

    rmse_near : RMSE over the quotes with |k| <= band (near the money).
    rmse_all : RMSE over every quote.
    band : √w_ATM, with w_ATM the market total variance at k = 0, linear in
        k between the neighbouring quotes.
    n_near : number of quotes with |k| <= band.
    """

    rmse_near: float
    rmse_all: float
    band: float
    n_near: int


def raw_svi(k, a, b, rho, m, s):
    """Raw SVI total implied variance.

    Parameters
    ----------
    k : array_like
        Log-moneyness ln(K/F).
    a, b, rho, m, s : float or array_like
        Raw SVI parameters; broadcast against k.

    Returns
    -------
    ndarray
        w(k) = a + b·(rho·(k - m) + √((k - m)² + s²)), total variance
        (decimal variance times years).
    """
    u = np.asarray(k, dtype=float) - m
    return a + b * (rho * u + np.sqrt(u * u + s * s))


def svi_constraints(a, b, rho, m, s):
    """Margins of the raw SVI constraints; each is >= 0 when its constraint holds.

    Parameters
    ----------
    a, b, rho, m, s : float or array_like
        Raw SVI parameters. m is unconstrained; it is accepted so that the
        five parameters of a fit can be passed as they are.

    Returns
    -------
    dict
        b : b (b >= 0)
        rho : 1 - |rho| (|rho| <= 1)
        s : s (s > 0 needs a margin above 0)
        w_min : a + b·s·√(1 - rho²), the minimum of w over k (w >= 0)
        lee : 2 - b·(1 + |rho|) (Lee's moment bound)
    """
    a, b, rho, s = (np.asarray(x, dtype=float) for x in (a, b, rho, s))
    root = np.sqrt(np.clip(1.0 - rho * rho, 0.0, None))
    return {
        "b": b[()],
        "rho": (1.0 - np.abs(rho))[()],
        "s": s[()],
        "w_min": (a + b * s * root)[()],
        "lee": (2.0 - b * (1.0 + np.abs(rho)))[()],
    }


def vega_weights(F, K, T, D, sigma):
    """Calibration weights of one expiry: Black-76 vega normalised to sum to 1.

    Parameters
    ----------
    F, K, T, D, sigma : array_like
        Forward, strikes, time to expiry in years, discount factor and the
        market implied vols (decimals), as in black_scholes.black76_vega.

    Returns
    -------
    ndarray
        ω_i = vega_i / Σ vega, the same shape as the broadcast inputs.

    Raises
    ------
    ValueError
        If any vega is not finite or negative, or they sum to 0.
    """
    vega = np.asarray(black76_vega(F, K, T, D, sigma), dtype=float)
    if not (np.isfinite(vega).all() and (vega >= 0).all() and vega.sum() > 0):
        raise ValueError("vega weights need finite, non-negative vegas with a positive sum")
    return vega / vega.sum()


def svi_objective(k, w, weights, a, b, rho, m, s):
    """Weighted squared error of a raw SVI slice in total variance.

    Parameters
    ----------
    k, w : array_like
        Log-moneyness and market total variance of the quotes.
    weights : array_like
        Weights ω of the quotes (vega_weights), used as given.
    a, b, rho, m, s : float
        Raw SVI parameters.

    Returns
    -------
    float
        Σ ω·(raw_svi(k) - w)².
    """
    residual = raw_svi(k, a, b, rho, m, s) - np.asarray(w, dtype=float)
    return float(np.asarray(weights, dtype=float) @ (residual * residual))


def _slice_arrays(k, w, weights):
    """Validated 1-D float arrays of one slice, with the weights normalised to sum to 1."""
    k, w, weights = (np.asarray(x, dtype=float) for x in (k, w, weights))
    if not (k.ndim == 1 and k.shape == w.shape == weights.shape):
        raise ValueError("k, w and weights must be 1-D arrays of the same length")
    if k.size < 5:
        raise ValueError(f"a slice needs at least 5 quotes, not {k.size}")
    if not (np.isfinite(k).all() and np.isfinite(w).all() and np.isfinite(weights).all()):
        raise ValueError("k, w and weights must be finite")
    if (w <= 0).any():
        raise ValueError("market total variance w must be positive")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    return k, w, weights / weights.sum()


def _inner_design(k, m, s):
    """Columns 1, √(y² + 1), y of the inner least squares in (a, c, d), with y = (k - m)/s."""
    y = (k - m) / s
    return np.column_stack([np.ones_like(y), np.sqrt(y * y + 1.0), y])


def _inner_feasible(a, c, d, s):
    """True when (a, c, d) satisfies every inner constraint exactly."""
    return c >= 0 and abs(d) <= c and c + abs(d) <= 2 * s and a + np.sqrt(max(c * c - d * d, 0.0)) >= 0


def _project_inner(a, c, d, s):
    """Move (a, c, d) onto the inner constraint set: c >= 0, |d| <= c, c + |d| <= 2s, then a + √(c² - d²) >= 0."""
    c = max(c, 0.0)
    d = min(max(d, -c), c)
    if c + abs(d) > 2 * s:
        shrink = 2 * s / (c + abs(d))
        c, d = c * shrink, d * shrink
    a = max(a, -np.sqrt(max(c * c - d * d, 0.0)))
    return a, c, d


def quasi_explicit_inner(k, w, weights, m, s):
    """Inner problem of the quasi-explicit calibration: the best (a, c, d) for fixed (m, s).

    Minimises Σ ω·(a + d·y + c·√(y² + 1) - w)², with y = (k - m)/s, subject to
    c >= 0, |d| <= c, c + |d| <= 2s and a + √(c² - d²) >= 0. The objective is
    a strictly convex quadratic (given three distinct k) and the constraint
    set is convex, so the minimiser is unique. The closed-form weighted least
    squares solution is tried first: when it satisfies every constraint it is
    that minimiser. Otherwise SLSQP solves the constrained problem with
    analytic gradients, from the feasible start (max(min w, 0), s/2, 0), on
    the objective divided by Σ ω·w² and in the variables (a, c, d)/v with
    v = Σ ω·w, and its solution is moved onto the constraint set (a
    correction of the order of SLSQP's constraint tolerance).

    SLSQP takes the w_min constraint in the equivalent cone form
    c >= √(d² + min(a, 0)²): with |d| <= c, a + √(c² - d²) >= 0 holds for
    a >= 0, and for a < 0 it is c² - d² >= a². The two forms describe the
    same set, but the gradient of √(c² - d²) is infinite where |d| = c
    (rho = ±1), which makes SLSQP's linearised constraints incompatible,
    while the cone form's gradient is bounded. When SLSQP reports failure it
    is restarted from its own point, for at most three runs.

    Parameters
    ----------
    k, w : array_like
        Log-moneyness and market total variance of the quotes (1-D).
    weights : array_like
        Non-negative weights; normalised here to sum to 1.
    m : float
        Translation, in units of k.
    s : float
        Curvature, > 0.

    Returns
    -------
    a, c, d : float
        The inner parameters, with c = b·s and d = rho·b·s.
    objective : float
        Σ ω·(w_fit - w)² with the normalised weights.

    Raises
    ------
    RuntimeError
        If SLSQP does not converge in three runs.
    """
    k, w, weights = _slice_arrays(k, w, weights)
    return _inner(k, w, weights, float(m), float(s))


def _cone(p):
    """The w_min constraint in cone form, c - √(d² + min(a, 0)²), at p = (a, c, d); >= 0 when it holds."""
    return p[1] - np.hypot(p[2], min(p[0], 0.0))


def _cone_jac(p):
    """Gradient of _cone in (a, c, d); (0, 1, 0) at the kink d = 0, a >= 0."""
    norm = np.hypot(p[2], min(p[0], 0.0))
    if norm == 0.0:
        return np.array([0.0, 1.0, 0.0])
    return np.array([-min(p[0], 0.0) / norm, 1.0, -p[2] / norm])


def _inner_slsqp(X, w, weights, m, s, strict):
    """SLSQP solution of the constrained inner problem, moved onto the constraint set (see quasi_explicit_inner)."""
    v = weights @ w  # the solver works in (a, c, d)/v, so that its variables are of order 1
    Xv, sv = X * v, s / v
    scale = weights @ (w * w)

    def objective(p):
        r = Xv @ p - w
        return weights @ (r * r) / scale, 2.0 * Xv.T @ (weights * r) / scale

    constraints = [
        {"type": "ineq", "fun": lambda p: p[1] - p[2], "jac": lambda p: np.array([0.0, 1.0, -1.0])},
        {"type": "ineq", "fun": lambda p: p[1] + p[2], "jac": lambda p: np.array([0.0, 1.0, 1.0])},
        {"type": "ineq", "fun": lambda p: 2 * sv - p[1] - p[2], "jac": lambda p: np.array([0.0, -1.0, -1.0])},
        {"type": "ineq", "fun": lambda p: 2 * sv - p[1] + p[2], "jac": lambda p: np.array([0.0, -1.0, 1.0])},
        {"type": "ineq", "fun": _cone, "jac": _cone_jac},
    ]
    x = np.array([max(w.min(), 0.0), s / 2, 0.0]) / v
    for _ in range(_SLSQP_RUNS):
        result = minimize(objective, x, jac=True, method="SLSQP", constraints=constraints, options=_SLSQP_OPTIONS)
        x = np.array(_project_inner(*(result.x * v), s)) / v
        if result.success:
            break
    else:
        if strict:
            raise RuntimeError(f"SLSQP did not converge at m = {m}, s = {s}: {result.message}")
    return _project_inner(*(x * v), s)


def _inner(k, w, weights, m, s, strict=True):
    """quasi_explicit_inner on validated arrays; strict=False keeps SLSQP's feasible point when it fails."""
    X = _inner_design(k, m, s)
    a, c, d = np.linalg.solve(X.T @ (weights[:, None] * X), X.T @ (weights * w))
    if not _inner_feasible(a, c, d, s):
        a, c, d = _inner_slsqp(X, w, weights, m, s, strict)
    r = X @ np.array([a, c, d]) - w
    return float(a), float(c), float(d), float(weights @ (r * r))


def _raw_from_inner(a, c, d, s):
    """Raw parameters (a, b, rho) from the inner (a, c, d): b = c/s, rho = d/c (0 when c = 0)."""
    return a, c / s, (d / c if c > 0 else 0.0)


def _bound_flags(m, rho, k_lo, k_hi):
    """Whether m lies on a bound of the quoted range and whether |rho| lies on 1, to 1e-8."""
    return bool(min(abs(m - k_lo), abs(m - k_hi)) <= M_BOUND_TOL), bool(1.0 - abs(rho) <= RHO_BOUND_TOL)


def fit_svi(k, w, weights):
    """Quasi-explicit raw SVI calibration of one expiry (Zeliade).

    The outer problem minimises the inner objective (quasi_explicit_inner),
    divided by Σ ω·w², over (m, s) with Nelder-Mead, m bounded to the quoted
    range [min k, max k] and s >= 1e-4, with xatol 1e-10 and fatol 1e-16. It
    runs from nine starts, m0 in {-1, 0, 1}·h clipped into the range and s0
    in {0.5, 1, 2}·h, with the scale h = √(Σ ω·w) (close to the ATM total
    vol, because vega weights peak at the money), and keeps the best. m is
    bounded because the vertex of the smile is identified only inside the
    quoted range: outside it, m extrapolates the unquoted wing.

    At a trial (m, s) of the search whose SLSQP solve fails three times, the
    search uses the feasible point SLSQP returned, whose objective bounds the
    inner minimum from above; the inner solve at the reported (m, s) must
    converge.

    Parameters
    ----------
    k, w : array_like
        Log-moneyness ln(K/F) and market total variance of the quotes (1-D,
        at least 5 quotes).
    weights : array_like
        Non-negative weights (vega_weights); normalised here to sum to 1.

    Returns
    -------
    SVIFit
        The raw parameters, the objective Σ ω·(w_fit - w)² with the
        normalised weights, and whether m and rho lie on their bounds.

    Raises
    ------
    ValueError
        On inputs that are not 1-D of one length, fewer than 5 quotes,
        values that are not finite, w not positive or invalid weights.
    RuntimeError
        If the inner solve at the reported (m, s) does not converge.
    """
    k, w, weights = _slice_arrays(k, w, weights)
    k_lo, k_hi = float(k.min()), float(k.max())
    scale = weights @ (w * w)
    h = np.sqrt(weights @ w)

    def outer(x):
        return _inner(k, w, weights, x[0], x[1], strict=False)[3] / scale

    best = None
    for m0 in _START_M:
        for s0 in _START_S:
            start = [min(max(m0 * h, k_lo), k_hi), max(s0 * h, S_MIN)]
            result = minimize(outer, start, method="Nelder-Mead", bounds=[(k_lo, k_hi), (S_MIN, None)],
                              options=_NM_OPTIONS)
            if best is None or result.fun < best.fun:
                best = result
    m, s = (float(x) for x in best.x)
    a, c, d, objective = _inner(k, w, weights, m, s)
    a, b, rho = _raw_from_inner(a, c, d, s)
    return SVIFit(a, b, rho, m, s, objective, *_bound_flags(m, rho, k_lo, k_hi))


def _direct_raw(x):
    """Raw (a, b, rho, m, s) from the box variables (alpha, beta, rho, m, s) of fit_svi_direct, and √(1 - rho²)."""
    alpha, beta, rho, m, s = x
    b = 2.0 * beta / (1.0 + abs(rho))
    root = np.sqrt(max(1.0 - rho * rho, 0.0))
    return alpha - b * s * root, b, rho, m, s, root


def fit_svi_direct(k, w, weights, n_starts=N_DIRECT_STARTS, seed=SEED_DIRECT):
    """Direct five-parameter raw SVI fit by L-BFGS-B from random starts (the cross-check of fit_svi).

    L-BFGS-B takes only box bounds, so the fit runs in box variables
    (alpha, beta, rho, m, s) with b = 2·beta/(1 + |rho|) and
    a = alpha - b·s·√(1 - rho²). Then alpha >= 0, 0 <= beta <= 1,
    -1 <= rho <= 1, min k <= m <= max k and s >= 1e-4 cover exactly the
    constraint set of fit_svi: alpha is the minimum of w and beta the share
    of Lee's bound used. The objective, divided by Σ ω·w², has an analytic
    gradient. Each start draws five uniforms from np.random.default_rng(seed),
    in the order alpha, beta, rho, m, s: alpha on [0, max w], beta on [0, 1],
    rho on [-1, 1], m on [min k, max k] and ln s on [ln 0.001, ln 2]. The best
    of the runs is returned.

    Parameters
    ----------
    k, w, weights
        As in fit_svi.
    n_starts : int
        Number of random starts.
    seed : int
        Seed of np.random.default_rng.

    Returns
    -------
    SVIFit
        As for fit_svi.
    """
    k, w, weights = _slice_arrays(k, w, weights)
    k_lo, k_hi = float(k.min()), float(k.max())
    scale = weights @ (w * w)

    def objective(x):
        alpha, beta, rho, m, s = x
        a, b, _, _, _, root = _direct_raw(x)
        u = k - m
        R = np.sqrt(u * u + s * s)
        r = a + b * (rho * u + R) - w
        db_dbeta = 2.0 / (1.0 + abs(rho))
        db_drho = -2.0 * beta * np.sign(rho) / (1.0 + abs(rho)) ** 2
        da_drho = -s * root * db_drho + b * s * rho / max(root, _ROOT_FLOOR)
        dw_db = rho * u + R
        jac = np.column_stack([
            np.ones_like(k),  # alpha
            -s * root * db_dbeta + dw_db * db_dbeta,  # beta, through a and b
            da_drho + dw_db * db_drho + b * u,  # rho, through a, b and directly
            -b * (rho + u / R),  # m
            -b * root + b * s / R,  # s, through a and directly
        ])
        return weights @ (r * r) / scale, 2.0 * jac.T @ (weights * r) / scale

    rng = np.random.default_rng(seed)
    draws = rng.uniform(size=(n_starts, 5))
    lo, hi = np.log(_DIRECT_S_RANGE)
    starts = np.column_stack([
        draws[:, 0] * w.max(),
        draws[:, 1],
        2.0 * draws[:, 2] - 1.0,
        k_lo + draws[:, 3] * (k_hi - k_lo),
        np.exp(lo + draws[:, 4] * (hi - lo)),
    ])
    bounds = [(0.0, None), (0.0, 1.0), (-1.0, 1.0), (k_lo, k_hi), (S_MIN, None)]
    best = None
    for start in starts:
        result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds, options=_LBFGSB_OPTIONS)
        if best is None or result.fun < best.fun:
            best = result
    a, b, rho, m, s, _ = (float(x) for x in _direct_raw(best.x))
    objective = svi_objective(k, w, weights, a, b, rho, m, s)
    return SVIFit(a, b, rho, m, s, objective, *_bound_flags(m, rho, k_lo, k_hi))


def svi_fit_errors(k, w, T, fit):
    """Fit errors of an SVI slice in implied vol, near the money and over every quote.

    The error of a quote is 100·(√(w_fit/T) - √(w/T)) vol points. Near the
    money means |k| <= √w_ATM, with w_ATM the market total variance at k = 0,
    linear in k between the neighbouring quotes. The RMSEs are unweighted.

    Parameters
    ----------
    k, w : array_like
        Log-moneyness and market total variance of the quotes (1-D, spanning
        k = 0).
    T : float
        Time to expiry in years.
    fit : SVIFit
        The fitted slice.

    Returns
    -------
    SVIErrors

    Raises
    ------
    ValueError
        If the quotes do not span k = 0.
    """
    k, w = np.asarray(k, dtype=float), np.asarray(w, dtype=float)
    if not k.min() <= 0.0 <= k.max():
        raise ValueError("the quotes must span k = 0 to read w_ATM off them")
    order = np.argsort(k)
    band = float(np.sqrt(np.interp(0.0, k[order], w[order])))
    w_fit = raw_svi(k, fit.a, fit.b, fit.rho, fit.m, fit.s)
    error = 100.0 * (np.sqrt(np.maximum(w_fit, 0.0) / T) - np.sqrt(w / T))
    near = np.abs(k) <= band
    return SVIErrors(
        float(np.sqrt(np.mean(error[near] ** 2))), float(np.sqrt(np.mean(error**2))), band, int(near.sum())
    )
