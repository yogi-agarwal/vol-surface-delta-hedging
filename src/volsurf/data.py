"""Data loading and caching utilities for SPY price, rate, and option chain data.

History columns
---------------
spy      : SPY adjusted close (total-return series, auto_adjust=True)
sigma_i  : ^VIX close / 100; forward-filled on SPY trading days when VIX is
           absent (rare holiday mismatches); count of filled days in manifest
r        : ^IRX close / 100; forward-filled on SPY trading days
Index    : DatetimeIndex of SPY trading days, UTC-normalised
"""

import json
import pathlib
from datetime import datetime, timezone

import pandas as pd
import yfinance

_REPO = pathlib.Path(__file__).resolve().parents[2]
_FROZEN = _REPO / "data" / "frozen"
_HISTORY = _FROZEN / "history.parquet"
_MANIFEST = _FROZEN / "manifest.json"
_START = "2021-06-01"


def _download_history(start: str) -> tuple[pd.DataFrame, dict]:
    """Download SPY, ^VIX and ^IRX from *start* to today and align on SPY days.

    Parameters
    ----------
    start : str
        ISO date string, e.g. '2021-06-01'.

    Returns
    -------
    df : pd.DataFrame
        Columns spy, sigma_i, r aligned on SPY trading days.
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
        DatetimeIndex of SPY trading days; columns spy, sigma_i, r (all float).
    """
    if not refresh and _HISTORY.exists():
        return pd.read_parquet(_HISTORY)

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

    return df
