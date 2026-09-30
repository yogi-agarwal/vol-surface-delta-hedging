"""Data loading and caching utilities for SPY price, rate, and option chain data.

History columns (as returned by load_history)
---------------------------------------------
spy      : SPY adjusted close (total-return series, auto_adjust=True)
sigma_i  : ^VIX close / 100; forward-filled on SPY trading days when VIX is
           absent (rare holiday mismatches); count of filled days in manifest
r        : continuously compounded rate irx_to_rate(d), where d = ^IRX
           close / 100 is the 13-week bill discount yield, forward-filled on
           SPY trading days
Index    : DatetimeIndex of SPY trading days at midnight, tz-naive

data/frozen/history.parquet stores d itself in column r; load_history
converts it on every load, so the frozen file never needs re-downloading.

OptionMetrics columns (as returned by load_optionmetrics)
---------------------------------------------------------
sigma_atm : mean of the call and put impl_volatility of the standardised
            30-day ATM-forward options
put_25d   : 30-day surface implied vol at delta -25 (put)
call_25d  : 30-day surface implied vol at delta 25 (call)
call_50d  : 30-day surface implied vol at delta 50 (call)
Index     : DatetimeIndex of OptionMetrics dates, named date, tz-naive

The OptionMetrics files are licensed WRDS extracts kept in data/raw/
(gitignored). They are read locally only, never downloaded or committed.
"""

import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance

_REPO = pathlib.Path(__file__).resolve().parents[2]
_FROZEN = _REPO / "data" / "frozen"
_HISTORY = _FROZEN / "history.parquet"
_MANIFEST = _FROZEN / "manifest.json"
_START = "2021-06-01"
_BILL_DAYS = 91  # ^IRX quotes the 13-week (91-day) bill

_RAW = _REPO / "data" / "raw"
_OM_STD = "om_spy_std_30d_2021_2025.csv"  # standardised options, 30-day ATM-forward
_OM_SURFACE = "om_spy_volsurf_2021_2025.csv"  # volatility surface, deltas by maturities
_OM_DAYS = 30
_OM_SURFACE_IVS = {"put_25d": (-25, "P"), "call_25d": (25, "C"), "call_50d": (50, "C")}


def irx_to_rate(d):
    """Continuously compounded rate from the 13-week T-bill discount yield.

    A 91-day bill quoted at discount yield d (ACT/360) costs 1 - d·91/360 per
    unit of face value, so the continuously compounded ACT/365 rate that
    grows that price to 1 is r = -ln(1 - d·91/360)/(91/365).

    Parameters
    ----------
    d : float, ndarray or pd.Series
        Discount yield as a decimal (^IRX close / 100).

    Returns
    -------
    Same type as d
        Rate r as a decimal, continuously compounded, ACT/365.
    """
    return -np.log1p(-d * (_BILL_DAYS / 360)) / (_BILL_DAYS / 365)


def _download_history(start: str) -> tuple[pd.DataFrame, dict]:
    """Download SPY, ^VIX and ^IRX from *start* to today and align on SPY days.

    Parameters
    ----------
    start : str
        ISO date string, e.g. '2021-06-01'.

    Returns
    -------
    df : pd.DataFrame
        Columns spy, sigma_i, r aligned on SPY trading days, with r holding
        the ^IRX discount yield d = close / 100 (as stored in the frozen file).
    ffill_counts : dict
        Number of days forward-filled per series: {"sigma_i": int, "r": int}.
    """
    kwargs = dict(progress=False, multi_level_index=False)

    spy = yfinance.download("SPY", start=start, auto_adjust=True, **kwargs)["Close"]
    vix_raw = yfinance.download("^VIX", start=start, auto_adjust=False, **kwargs)["Close"]
    irx_raw = yfinance.download("^IRX", start=start, auto_adjust=False, **kwargs)["Close"]

    for name, s in [("spy", spy), ("vix_raw", vix_raw), ("irx_raw", irx_raw)]:
        if not (isinstance(s, pd.Series) and s.ndim == 1):
            raise ValueError(f"{name} must be a 1-D float Series; got {type(s)}")

    vix_aligned = vix_raw.reindex(spy.index)
    irx_aligned = irx_raw.reindex(spy.index)

    ffill_counts = {
        "sigma_i": int(vix_aligned.isna().sum()),
        "r": int(irx_aligned.isna().sum()),
    }

    sigma_i = vix_aligned.ffill() / 100
    r = irx_aligned.ffill() / 100

    df = pd.DataFrame({"spy": spy, "sigma_i": sigma_i, "r": r})
    return df, ffill_counts


def load_history(refresh: bool = False) -> pd.DataFrame:
    """Load SPY/VIX/IRX history from the frozen Parquet file, downloading if needed.

    Parameters
    ----------
    refresh : bool
        If True, re-download from yfinance and overwrite the frozen file.
        If False (default), read from data/frozen/history.parquet when it exists.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex of SPY trading days; columns spy, sigma_i, r (all float),
        with r converted from the stored discount yield by irx_to_rate.
    """
    if not refresh and _HISTORY.exists():
        df = pd.read_parquet(_HISTORY)
    else:
        df, ffill_counts = _download_history(_START)

        _FROZEN.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_HISTORY)

        manifest = {
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "tickers": ["SPY", "^VIX", "^IRX"],
            "yfinance_version": yfinance.__version__,
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "row_count": len(df),
            "ffill_counts": ffill_counts,
        }
        _MANIFEST.write_text(json.dumps(manifest, indent=2))

    df["r"] = irx_to_rate(df["r"])
    return df


def load_optionmetrics(raw_dir=None):
    """Load the licensed OptionMetrics 30-day SPY implied vols from local files.

    The standardised options file gives one 30-day ATM-forward call and put
    per date; the volatility surface file is read at days == 30 only. Only
    local files are read: nothing is ever downloaded.

    Parameters
    ----------
    raw_dir : str or pathlib.Path, optional
        Directory holding om_spy_std_30d_2021_2025.csv and
        om_spy_volsurf_2021_2025.csv. Defaults to data/raw/ of the repository.

    Returns
    -------
    pd.DataFrame or None
        None if either file is absent. Otherwise one row per date (sorted
        DatetimeIndex named date, tz-naive) with columns sigma_atm, put_25d,
        call_25d and call_50d, annualised implied vols as decimals.
        sigma_atm = (call IV + put IV)/2 and is NaN when either is missing;
        a date present in only one file has NaN in the other file's columns.

    Raises
    ------
    ValueError
        If the standardised file repeats a (date, cp_flag) pair, or the
        surface a (date, delta, cp_flag) triple, at days == 30.
    """
    raw = _RAW if raw_dir is None else pathlib.Path(raw_dir)
    std_path, surface_path = raw / _OM_STD, raw / _OM_SURFACE
    if not (std_path.exists() and surface_path.exists()):
        return None

    cols = ["date", "days", "cp_flag", "impl_volatility"]
    std = pd.read_csv(std_path, usecols=cols, parse_dates=["date"])
    std = std[std["days"] == _OM_DAYS]
    if std.duplicated(["date", "cp_flag"]).any():
        raise ValueError(f"{_OM_STD} repeats a (date, cp_flag) pair at days = {_OM_DAYS}")
    iv = std.pivot(index="date", columns="cp_flag", values="impl_volatility")
    columns = {"sigma_atm": (iv["C"] + iv["P"]) / 2}

    surface = pd.read_csv(surface_path, usecols=cols + ["delta"], parse_dates=["date"])
    surface = surface[surface["days"] == _OM_DAYS]
    if surface.duplicated(["date", "delta", "cp_flag"]).any():
        raise ValueError(f"{_OM_SURFACE} repeats a (date, delta, cp_flag) triple at days = {_OM_DAYS}")
    for name, (delta, flag) in _OM_SURFACE_IVS.items():
        rows = surface[(surface["delta"] == delta) & (surface["cp_flag"] == flag)]
        columns[name] = rows.set_index("date")["impl_volatility"]

    out = pd.concat(columns, axis=1).sort_index()
    out.index.name = "date"
    return out
