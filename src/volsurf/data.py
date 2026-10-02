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
The ^IRX closes after the history's last date, which chain snapshots taken
later need, are frozen as d in data/frozen/irx_closes.csv (load_irx_closes).

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

Chain columns (as returned by clean_chain and load_chain)
---------------------------------------------------------
Every contract of the selected expiries, one row each, with the raw snapshot
columns (contractSymbol, strike, bid, ask, lastPrice, volume, openInterest,
lastTradeDate, option_type 'call' or 'put', fetch_utc, spot_start, spot_end)
and
expiry : expiry date, tz-naive midnight
mid    : (bid + ask)/2
T      : years from the expiry's fetch_utc to 16:00 America/New_York on the
         expiry date, ACT/365
rate   : continuously compounded rate of the expiry (the flat ^IRX rate,
         the same for every expiry, unless clean_chain is given a curve r(T))
D      : discount factor exp(-rate·T)
F      : parity forward of the expiry (fixed-D fit, see parity_forward)
k      : log-moneyness ln(K/F)
status : 'kept', or the first rule that removed the contract: 'zero bid',
         'spread', 'open interest', 'in the money' (against S_bar, the spot
         at the quote time: puts with K >= S_bar, calls with K < S_bar).
         Stage 2c relabels kept contracts without an implied vol 'no IV' in
         the notebook's copy; the frozen file never holds that label

data/frozen/chain_YYYYMMDD.parquet stores this table, so Table 1 rebuilds
from data/frozen/ alone. Snapshots collected after 2026-09-30 also carry
spot_bar_start, spot_bar_end (closes of the latest SPY 1-minute bar at the
start and end of the pull) and spot_bar_start_utc, spot_bar_end_utc (those
bars' start times).

Minute bars (as returned by load_minute_bars)
---------------------------------------------
Open, High, Low, Close : unadjusted prices of each 1-minute bar (SPY in
                         dollars, regular session; ^VIX in index points,
                         global trading hours included)
Volume                 : shares traded in the bar (0 for ^VIX)
Index                  : DatetimeIndex of bar starts in UTC, named bar_start

data/frozen/spy_1m_YYYYMMDD.parquet stores one day's SPY bars; snapshot_spot_bar
matches them to a chain snapshot. A snapshot collected from 2026-10-01 also
has the bars saved with its pull, data/raw/spy_1m_YYYYMMDDTHHMMSSZ.parquet
(same columns and index), which snapshot_spot_bar reads first; load_chain
freezes a copy under the same name in data/frozen/.
data/frozen/vix_1m_YYYYMMDD.parquet stores one day's ^VIX bars, which
snapshot_vix_bar reads for VIX at a snapshot's quote time.

Treasury curve (as returned by load_treasury_curve)
---------------------------------------------------
DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3 : FRED Treasury constant-maturity
                                          yields in percent, as published
Index                                    : DatetimeIndex of dates, named date

data/frozen/ust_curve_YYYYMMDD.csv stores the 14 days up to a snapshot date;
treasury_rates turns one date into continuously compounded rates.
"""

import json
import pathlib
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance

_REPO = pathlib.Path(__file__).resolve().parents[2]
_FROZEN = _REPO / "data" / "frozen"
_HISTORY = _FROZEN / "history.parquet"
_MANIFEST = _FROZEN / "manifest.json"
_START = "2021-06-01"
_BILL_DAYS = 91  # ^IRX quotes the 13-week (91-day) bill
_IRX_CLOSES = "irx_closes.csv"  # ^IRX closes after the end of the frozen history, for later chain snapshots

_NY = ZoneInfo("America/New_York")
_YEAR_SECONDS = 365 * 86400  # ACT/365
_EXPIRY_HOUR = 16  # options expire at the 16:00 New York close
_MIN_T_DAYS = 7
_SLICES = (8, 12)  # allowed number of selected expiries
_LONG_MONTHS = (1, 3, 6, 9, 12)  # LEAPS (January) and quarterly months kept beyond 1 year
_BAND = 0.05  # |ln(K/S)| band of the parity fits
_MIN_PAIRS = 6  # fixed-D forward fit, -0.05 <= ln(K/S) <= 0
_MIN_PAIRS_FREE = 8  # free-slope diagnostics, two-sided and one-sided
_ONE_SIDED_SCALE = 0.1  # one-sided diagnostic band: -max(0.05, 0.1·√T) <= ln(K/S) <= 0
_MAX_SPREAD = 0.25  # largest (ask - bid)/mid kept
_MIN_OPEN_INTEREST = 10
_RATE_MAX_AGE_DAYS = 5  # the ^IRX close may precede the snapshot by at most this many days
_SNAPSHOT_NAME = re.compile(r"spy_chain_(\d{8})T\d{6}Z\.parquet")
_BAR = pd.Timedelta(minutes=1)  # Yahoo 1-minute bars, labelled by their opening minute
_BAR_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_UST_TENORS = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0}  # years
_UST_WINDOW_DAYS = 14  # calendar days of the curve downloaded up to the snapshot date

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

        _update_manifest({
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "tickers": ["SPY", "^VIX", "^IRX"],
            "yfinance_version": yfinance.__version__,
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "row_count": len(df),
            "ffill_counts": ffill_counts,
        })

    df["r"] = irx_to_rate(df["r"])
    return df


def load_irx_closes(refresh=False, end=None):
    """^IRX closes after the end of the frozen history, as continuously compounded rates.

    A chain snapshot is priced at the rate of the last ^IRX close strictly
    before its own date (DESIGN.md section 5). The frozen history ends on its
    download date, and refreshing it would add Stage 5 windows, so the closes
    that later snapshots need are frozen separately in
    data/frozen/irx_closes.csv, as the discount yield d = close / 100 (like
    the history), with an entry under "irx_closes" in
    data/frozen/manifest.json. Only refresh=True downloads: the closes from
    the day after the history's last date up to, but not including, end, so
    an unfinished session is never stored.

    Parameters
    ----------
    refresh : bool
        If True, download the closes and overwrite the frozen file.
    end : date-like, optional
        First date not downloaded; today in New York by default.

    Returns
    -------
    pd.Series
        irx_to_rate(d) on a tz-naive DatetimeIndex named date, named r;
        empty when the frozen file does not exist and refresh is False.

    Raises
    ------
    ValueError
        If a download returns no close in the range.
    """
    path = _FROZEN / _IRX_CLOSES
    if not refresh:
        if not path.exists():
            return pd.Series(dtype=float, name="r", index=pd.DatetimeIndex([], name="date"))
        d = pd.read_csv(path, index_col="date", parse_dates=["date"])["d"]
        return irx_to_rate(d).rename("r")

    last = load_history().index[-1]
    end = pd.Timestamp(datetime.now(_NY).date()) if end is None else pd.Timestamp(end).normalize()
    close = yfinance.download(
        "^IRX", start=f"{last + pd.Timedelta(days=1):%Y-%m-%d}", end=f"{end:%Y-%m-%d}", auto_adjust=False,
        progress=False, multi_level_index=False,
    )["Close"]
    dates = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    d = pd.Series(close.to_numpy() / 100, index=dates.rename("date"), name="d").dropna()
    d = d[(d.index > last) & (d.index < end)]
    if d.empty:
        raise ValueError(f"no ^IRX close after {last.date()} and before {end.date()}")
    _FROZEN.mkdir(parents=True, exist_ok=True)
    d.to_csv(path)
    _update_manifest({"irx_closes": {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "file": path.name,
        "ticker": "^IRX",
        "yfinance_version": yfinance.__version__,
        "after_history": str(last.date()),
        "start_date": str(d.index[0].date()),
        "end_date": str(d.index[-1].date()),
        "row_count": len(d),
    }})
    return irx_to_rate(d).rename("r")


def _rate_history():
    """The frozen history's rates followed by the later frozen ^IRX closes, one series in date order."""
    r = load_history()["r"]
    later = load_irx_closes()
    return pd.concat([r, later[later.index > r.index[-1]]])


def _update_manifest(entries):
    """Merge *entries* into data/frozen/manifest.json, keeping every other key.

    A dict value merges one level deep into an existing dict under the same
    key (so a new chain entry joins the earlier ones); any other value
    replaces the old one.
    """
    manifest = json.loads(_MANIFEST.read_text()) if _MANIFEST.exists() else {}
    for key, value in entries.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key] = {**manifest[key], **value}
        else:
            manifest[key] = value
    _MANIFEST.write_text(json.dumps(manifest, indent=2))


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


def monthly_expiries(expiries):
    """Flag the monthly expiry of each month among the listed expiries.

    The monthly expiry is the third Friday of the month, or the Thursday
    before it when that Friday is not listed (an exchange holiday, such as
    Good Friday or Juneteenth observed).

    Parameters
    ----------
    expiries : array_like of date-like
        Every listed expiry date of the snapshot (str 'YYYY-MM-DD' or
        timestamps; any time of day is ignored).

    Returns
    -------
    ndarray of bool
        True where the expiry is its month's monthly expiry.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(expiries))).normalize()
    first = dates.to_period("M").to_timestamp()
    third_friday = first + pd.to_timedelta((4 - first.weekday.to_numpy()) % 7 + 14, unit="D")
    holiday_thursday = (dates == third_friday - pd.Timedelta(days=1)) & ~third_friday.isin(dates)
    return np.asarray((dates == third_friday) | holiday_thursday)


def time_to_expiry(expiry, fetch_utc):
    """Time from a fetch timestamp to the 16:00 New York close on the expiry date.

    Parameters
    ----------
    expiry : date-like or array_like of date-like
        Expiry dates (any time of day is ignored).
    fetch_utc : datetime-like or array_like of datetime-like
        Fetch timestamps, tz-aware (tz-naive values are read as UTC).
        Broadcasts against expiry.

    Returns
    -------
    float or ndarray
        T in years, (expiry 16:00 America/New_York - fetch_utc)/365 days,
        with daylight saving handled by the time zone. A float when both
        inputs are scalars.
    """
    close = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(expiry))).normalize()
    close = (close + pd.Timedelta(hours=_EXPIRY_HOUR)).tz_localize(_NY)
    fetch = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(fetch_utc), utc=True))
    seconds = (close.as_unit("ns").asi8 - fetch.as_unit("ns").asi8) / 1e9
    T = seconds / _YEAR_SECONDS
    return float(T[0]) if np.ndim(expiry) == 0 and np.ndim(fetch_utc) == 0 else T


def select_expiries(expiry, T):
    """Select the expiry slices of the surface.

    Keeps the monthly expiries (see monthly_expiries) with 7 days <= T <= 1
    year, and beyond 1 year the quarterly (March, June, September, December)
    and LEAPS (January) monthlies.

    Parameters
    ----------
    expiry : array_like of date-like
        Every listed expiry date of the snapshot, each once.
    T : array_like of float
        Time to each expiry in years (time_to_expiry), same shape.

    Returns
    -------
    ndarray of bool
        True for the selected expiries.

    Raises
    ------
    ValueError
        If the selection does not give 8 to 12 slices.
    """
    T = np.asarray(T, dtype=float)
    month = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(expiry))).month.to_numpy()
    keep = monthly_expiries(expiry) & (T >= _MIN_T_DAYS / 365) & ((T <= 1.0) | np.isin(month, _LONG_MONTHS))
    lo, hi = _SLICES
    if not lo <= keep.sum() <= hi:
        raise ValueError(f"expiry selection gives {keep.sum()} slices; {lo} to {hi} are required")
    return keep


def _quote_arrays(K, call_bid, call_ask, put_bid, put_ask):
    """Float arrays of one expiry's pairs: K, C_mid - P_mid and the two half-spreads."""
    K, cb, ca, pb, pa = (np.asarray(x, dtype=float) for x in (K, call_bid, call_ask, put_bid, put_ask))
    return K, (cb + ca) / 2 - (pb + pa) / 2, (ca - cb) / 2, (pa - pb) / 2


def parity_forward(K, call_bid, call_ask, put_bid, put_ask, D=None):
    """Forward and discount factor from put-call parity on one expiry's pairs.

    Parity for European options gives C - P = D·(F - K). Mids are fitted by
    weighted least squares with weights 1/(h_C² + h_P²), where h is a
    quote's half-spread (ask - bid)/2.

    Parameters
    ----------
    K : array_like
        Strikes of the call-put pairs.
    call_bid, call_ask, put_bid, put_ask : array_like
        Quotes at each strike; every ask must exceed its bid.
    D : float, optional
        Known discount factor to expiry. If given, the slope is held at -D
        and F = Σ w·(K + (C_mid - P_mid)/D) / Σ w in closed form. If None,
        the slope is free: D = -slope and F = intercept/D.

    Returns
    -------
    F, D : float
        Forward price (currency units of K) and discount factor.

    Raises
    ------
    ValueError
        If there are fewer than 2 pairs or a half-spread is not positive.
    """
    K, y, h_call, h_put = _quote_arrays(K, call_bid, call_ask, put_bid, put_ask)
    if K.size < 2:
        raise ValueError(f"parity fit needs at least 2 pairs, got {K.size}")
    if not (np.all(h_call > 0) and np.all(h_put > 0)):
        raise ValueError("every quote in a parity fit needs ask > bid")
    w = 1 / (h_call**2 + h_put**2)
    if D is not None:
        return float(np.sum(w * (K + y / D)) / np.sum(w)), float(D)
    K_bar = np.sum(w * K) / np.sum(w)  # centring the regressor keeps the fit well conditioned
    sw = np.sqrt(w)
    (level, slope), *_ = np.linalg.lstsq(np.column_stack([sw, sw * (K - K_bar)]), sw * y, rcond=None)
    D = -slope  # y = D·(F - K_bar) - D·(K - K_bar)
    return float(K_bar + level / D), float(D)


def parity_residuals(K, call_bid, call_ask, put_bid, put_ask, F, D):
    """Parity residuals of one expiry's pairs and their combined half-spreads.

    Parameters
    ----------
    K, call_bid, call_ask, put_bid, put_ask : array_like
        Strikes and quotes of the call-put pairs.
    F, D : float
        Forward and discount factor of the expiry.

    Returns
    -------
    residual : ndarray
        (C_mid - P_mid) - D·(F - K), in the currency units of K.
    half_spread : ndarray
        h_C + h_P, the half-width of the synthetic C - P market, which runs
        from C_bid - P_ask to C_ask - P_bid. Parity holds within the market
        where |residual| <= half_spread.
    """
    K, y, h_call, h_put = _quote_arrays(K, call_bid, call_ask, put_bid, put_ask)
    return y - D * (F - K), h_call + h_put


def filter_quotes(bid, ask, open_interest):
    """Apply the quote filters in order and label each contract.

    Rules, in order: zero bid (bid not above 0); spread, (ask - bid)/mid
    above 0.25; open interest below 10. A contract is labelled with the
    first rule it fails.

    Parameters
    ----------
    bid, ask : array_like
        Quotes in currency units.
    open_interest : array_like
        Open interest in contracts.

    Returns
    -------
    ndarray of str
        'kept', 'zero bid', 'spread' or 'open interest' per contract.
    """
    bid, ask, oi = (np.asarray(x, dtype=float) for x in (bid, ask, open_interest))
    with np.errstate(divide="ignore", invalid="ignore"):
        wide = ~((ask - bid) / ((bid + ask) / 2) <= _MAX_SPREAD)  # a missing ask counts as wide
    rules = [~(bid > 0), wide, ~(oi >= _MIN_OPEN_INTEREST)]
    return np.select(rules, ["zero bid", "spread", "open interest"], default="kept")


def _parity_pairs(rows):
    """Valid call-put pairs of one expiry, sorted by strike.

    A pair is valid when both the call and the put have bid > 0 and
    ask > bid. Returns the arrays (K, call_bid, call_ask, put_bid, put_ask).
    """
    calls = rows[rows["option_type"] == "call"].set_index("strike")[["bid", "ask"]]
    puts = rows[rows["option_type"] == "put"].set_index("strike")[["bid", "ask"]]
    pairs = calls.join(puts, how="inner", lsuffix="_c", rsuffix="_p").sort_index()
    valid = (
        (pairs["bid_c"] > 0) & (pairs["ask_c"] > pairs["bid_c"])
        & (pairs["bid_p"] > 0) & (pairs["ask_p"] > pairs["bid_p"])
    )
    pairs = pairs[valid]
    quotes = [pairs[c].to_numpy(dtype=float) for c in ["bid_c", "ask_c", "bid_p", "ask_p"]]
    return pairs.index.to_numpy(dtype=float), *quotes


def _selected_expiries(raw):
    """The expiries of a raw snapshot that select_expiries keeps, from each expiry's own fetch_utc."""
    listed = raw.groupby("expiry")["fetch_utc"].first()
    return listed.index[select_expiries(listed.index, time_to_expiry(listed.index, listed.to_numpy()))]


def clean_chain(raw, rate, spot):
    """Select expiries, fit parity forwards and label every contract.

    Steps: T per expiry from its own fetch_utc; expiry selection
    (select_expiries); per expiry, D = exp(-r·T) and F from parity_forward
    with D fixed, over the valid pairs with -0.05 <= ln(K/S) <= 0, where S is
    spot_start; filter_quotes on every contract; then out-of-the-money
    selection at the spot of the quote time S_bar (puts with K < S_bar and
    calls with K >= S_bar are kept, the others are labelled 'in the money');
    k = ln(K/F).

    The boundary is S_bar, not F: a put with S_bar <= K < F is in the money
    against the spot, so its American early-exercise premium inflates its
    implied vol, and the call at the same strike is out of the money.

    Parameters
    ----------
    raw : pd.DataFrame
        A chain snapshot as written by scripts/collect_chain.py: one row per
        contract with strike, bid, ask, openInterest, expiry, option_type
        ('call' or 'put'), fetch_utc (one per expiry) and spot_start (one
        value), plus any other columns, which are carried through.
    rate : float or callable
        Continuously compounded rate r as a decimal, used for every expiry;
        or a function of T (years, ndarray) that returns each expiry's rate
        r(T), such as a Treasury curve through interp_rate.
    spot : float
        S_bar, the spot at the quote time in currency units (the close of the
        SPY 1-minute bar nearest the latest option trade, as recorded by
        snapshot_spot_bar): the boundary of the out-of-the-money selection.

    Returns
    -------
    pd.DataFrame
        Every contract of the selected expiries, sorted by expiry, type and
        strike, with the columns described in the module docstring.

    Raises
    ------
    ValueError
        On duplicate (expiry, option_type, strike) rows, an unknown
        option_type, more than one fetch_utc per expiry or more than one
        spot_start, a spot that is not finite and positive, a selection
        outside 8 to 12 slices, or an expiry with fewer than 6 valid pairs in
        the fit band.
    """
    if raw.duplicated(["expiry", "option_type", "strike"]).any():
        raise ValueError("the snapshot repeats an (expiry, option_type, strike) row")
    if not raw["option_type"].isin(["call", "put"]).all():
        raise ValueError("option_type must be 'call' or 'put'")
    if raw.groupby("expiry")["fetch_utc"].nunique().max() > 1:
        raise ValueError("each expiry needs a single fetch_utc")
    if raw["spot_start"].nunique() != 1:
        raise ValueError("the snapshot needs a single spot_start")
    if not (np.isfinite(spot) and spot > 0):
        raise ValueError(f"the out-of-the-money boundary needs a finite positive spot, not {spot}")
    band_spot = float(raw["spot_start"].iloc[0])

    chain = raw[raw["expiry"].isin(_selected_expiries(raw))].copy()
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    chain = chain.sort_values(["expiry", "option_type", "strike"], ignore_index=True)

    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain["T"] = time_to_expiry(chain["expiry"], chain["fetch_utc"])
    chain["rate"] = rate(chain["T"].to_numpy()) if callable(rate) else rate
    chain["D"] = np.exp(-chain["rate"] * chain["T"])
    forwards = {}
    for expiry, rows in chain.groupby("expiry"):
        K, *quotes = _parity_pairs(rows)
        x = np.log(K / band_spot)
        band = (x >= -_BAND) & (x <= 0)
        if band.sum() < _MIN_PAIRS:
            raise ValueError(
                f"expiry {expiry.date()}: {band.sum()} valid pairs with {-_BAND} <= ln(K/S) <= 0, "
                f"at least {_MIN_PAIRS} required"
            )
        forwards[expiry], _ = parity_forward(K[band], *(q[band] for q in quotes), D=rows["D"].iloc[0])
    chain["F"] = chain["expiry"].map(forwards)
    chain["k"] = np.log(chain["strike"] / chain["F"])

    status = filter_quotes(chain["bid"], chain["ask"], chain["openInterest"])
    is_call = (chain["option_type"] == "call").to_numpy()
    in_the_money = np.where(is_call, chain["strike"] < spot, chain["strike"] >= spot)
    chain["status"] = np.where((status == "kept") & in_the_money, "in the money", status)
    return chain


def chain_summary(chain, spot=None):
    """Table 1 per expiry, computed from a cleaned chain alone.

    Parameters
    ----------
    chain : pd.DataFrame
        Output of clean_chain or load_chain, or a copy whose kept contracts
        without an implied vol are relabelled 'no IV' (Stage 2c).
    spot : float, optional
        Spot S of the implied carry q, in currency units (snapshot_spot_bar
        gives the spot at the quote time). Defaults to spot_start. The parity
        bands always use spot_start, as clean_chain does.

    Returns
    -------
    pd.DataFrame
        One row per expiry (index expiry) with
        T_days : T in calendar days
        contracts : contracts of the expiry in the snapshot
        zero_bid, spread, open_interest, in_the_money : removals per rule
        no_IV : kept contracts that the Stage 2c no-IV filter removed
            (status 'no IV'); 0 on a chain from clean_chain or load_chain
        kept_puts, kept_calls, kept : contracts kept
        K_min, K_max : strike range of the kept contracts
        pairs, pairs_upper : valid pairs with -0.05 <= ln(K/S) <= 0 (the fit)
            and with 0 < ln(K/S) <= 0.05 (excluded from the fit)
        F, D, rate : forward, discount factor and rate used downstream
        q : implied carry rate - ln(F/S)/T, as a decimal, with S = spot
        within, within_upper : share of those pairs whose parity residual
            lies within the combined half-spread (NaN without pairs)
        F_free, D_free, r_free : free-slope parity fit over |ln(K/S)| <= 0.05
            and its implied rate -ln(D_free)/T (NaN with fewer than 8 pairs);
            a diagnostic only
        pairs_one_sided, D_one_sided, r_one_sided : valid pairs with
            -max(0.05, 0.1·√T) <= ln(K/S) <= 0, where the put is out of the
            money, their free-slope parity D and its implied rate
            -ln(D_one_sided)/T (NaN with fewer than 8 pairs); a diagnostic only
    """
    band_spot = float(chain["spot_start"].iloc[0])
    carry_spot = band_spot if spot is None else float(spot)
    rules = ["zero bid", "spread", "open interest", "in the money", "no IV"]
    rows = []
    for expiry, g in chain.groupby("expiry"):
        T, F, D, rate = (float(g[c].iloc[0]) for c in ["T", "F", "D", "rate"])
        kept = g[g["status"] == "kept"]
        removed = g["status"].value_counts()
        K, *quotes = _parity_pairs(g)
        x = np.log(K / band_spot)
        fit, upper, both = (x >= -_BAND) & (x <= 0), (x > 0) & (x <= _BAND), np.abs(x) <= _BAND
        one_sided = (x >= -max(_BAND, _ONE_SIDED_SCALE * np.sqrt(T))) & (x <= 0)
        residual, half_spread = parity_residuals(K, *quotes, F, D)
        within = np.abs(residual) <= half_spread
        F_free, D_free, D_one_sided = np.nan, np.nan, np.nan
        if both.sum() >= _MIN_PAIRS_FREE:
            F_free, D_free = parity_forward(K[both], *(q[both] for q in quotes))
        if one_sided.sum() >= _MIN_PAIRS_FREE:
            _, D_one_sided = parity_forward(K[one_sided], *(q[one_sided] for q in quotes))
        rows.append({
            "expiry": expiry,
            "T_days": 365 * T,
            "contracts": len(g),
            **{rule.replace(" ", "_"): int(removed.get(rule, 0)) for rule in rules},
            "kept_puts": int((kept["option_type"] == "put").sum()),
            "kept_calls": int((kept["option_type"] == "call").sum()),
            "kept": len(kept),
            "K_min": kept["strike"].min(),
            "K_max": kept["strike"].max(),
            "pairs": int(fit.sum()),
            "pairs_upper": int(upper.sum()),
            "F": F,
            "D": D,
            "rate": rate,
            "q": rate - np.log(F / carry_spot) / T,
            "within": within[fit].mean() if fit.any() else np.nan,
            "within_upper": within[upper].mean() if upper.any() else np.nan,
            "F_free": F_free,
            "D_free": D_free,
            "r_free": -np.log(D_free) / T,
            "pairs_one_sided": int(one_sided.sum()),
            "D_one_sided": D_one_sided,
            "r_one_sided": -np.log(D_one_sided) / T,
        })
    return pd.DataFrame(rows).set_index("expiry")


def near_money_spread(chain, band=_BAND):
    """Median relative bid-ask spread of the near-the-money kept quotes of a cleaned chain.

    The metric of the primary-snapshot rule (DESIGN.md section 5): the median
    of (ask - bid)/mid over the contracts with status kept (after the Stage 2b
    filters and the out-of-the-money selection, before the no-IV filter), all
    expiries pooled, with |ln(K/S)| <= band and S the spot quote spot_start,
    as in the fit bands.

    Parameters
    ----------
    chain : pd.DataFrame
        Output of clean_chain or load_chain.
    band : float, default 0.05
        Half-width of the near-the-money band in ln(K/S).

    Returns
    -------
    float
        The median relative spread, as a decimal.

    Raises
    ------
    ValueError
        If no kept quote lies inside the band.
    """
    kept = chain[chain["status"] == "kept"]
    near = np.abs(np.log(kept["strike"] / kept["spot_start"])) <= band
    if not near.any():
        raise ValueError("no kept quote inside the near-the-money band")
    spread = (kept["ask"] - kept["bid"]) / kept["mid"]
    return float(spread[near].median())


def primary_snapshot(spreads):
    """The primary snapshot: the lower median near-the-money spread, a tie going to the later snapshot.

    Parameters
    ----------
    spreads : dict of str to float
        Raw snapshot file name (spy_chain_YYYYMMDDTHHMMSSZ.parquet) of each
        eligible snapshot and its near_money_spread.

    Returns
    -------
    str
        The file name of the primary.

    Raises
    ------
    ValueError
        If spreads is empty, a name does not match, or a spread is not finite.
    """
    if not spreads:
        raise ValueError("no eligible snapshot")
    best = None
    for name in sorted(spreads):  # the timestamped names sort in time
        if _SNAPSHOT_NAME.fullmatch(name) is None:
            raise ValueError(f"{name} is not a spy_chain_YYYYMMDDTHHMMSSZ.parquet snapshot")
        if not np.isfinite(spreads[name]):
            raise ValueError(f"the spread of {name} is not finite")
        if best is None or spreads[name] <= spreads[best]:  # <=: equal spreads go to the later snapshot
            best = name
    return best


def _snapshot_rate(r, snapshot_date):
    """Last rate strictly before the snapshot date, and its date.

    Parameters
    ----------
    r : pd.Series
        Continuously compounded rates on a DatetimeIndex (load_history()['r']).
    snapshot_date : pd.Timestamp
        Snapshot date in New York, tz-naive midnight.

    Returns
    -------
    rate : float
    rate_date : pd.Timestamp

    Raises
    ------
    ValueError
        If no rate precedes the snapshot date by at most 5 calendar days.
    """
    before = r[r.index < snapshot_date]
    if before.empty or (snapshot_date - before.index[-1]).days > _RATE_MAX_AGE_DAYS:
        raise ValueError(f"no ^IRX close within {_RATE_MAX_AGE_DAYS} days before {snapshot_date.date()}")
    return float(before.iloc[-1]), before.index[-1]


def load_chain(snapshot, refresh=False):
    """Load a cleaned chain from data/frozen/, building it from its raw snapshot if needed.

    The frozen file is data/frozen/chain_YYYYMMDD.parquet, dated by the
    snapshot's file name. When it exists and refresh is False it is read and
    the raw snapshot is not needed. Otherwise the raw snapshot is cleaned
    with clean_chain at the rate of the last ^IRX close strictly before the
    snapshot's New York date (from load_history, then the later closes of
    load_irx_closes) and at S_bar, the spot at the quote time: the close of
    the SPY 1-minute bar nearest the latest option trade of the selected
    expiries. The bars come from the sources snapshot_spot_bar reads (the
    bars saved with the pull, then their frozen copy, then the frozen day
    file), and the day is downloaded only when none exists. Bars saved with
    the pull are first copied into data/frozen/ under the same name, with an
    entry under "minute_bars" in the manifest, so a fresh install reproduces
    S_bar without data/raw/. The chain is written to the frozen file and
    recorded under "chains" in data/frozen/manifest.json, with the spot_bar
    and spot_bar_fetch entries that snapshot_spot_bar returns.

    Parameters
    ----------
    snapshot : str or pathlib.Path
        Raw snapshot spy_chain_YYYYMMDDTHHMMSSZ.parquet, as a path or as a
        bare file name in data/raw/.
    refresh : bool
        If True, rebuild the frozen file from the raw snapshot.

    Returns
    -------
    pd.DataFrame
        The cleaned chain (columns in the module docstring).

    Raises
    ------
    ValueError
        If the file name does not match spy_chain_YYYYMMDDTHHMMSSZ.parquet.
    """
    path = pathlib.Path(snapshot)
    match = _SNAPSHOT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path.name} is not a spy_chain_YYYYMMDDTHHMMSSZ.parquet snapshot")
    if path.parent == pathlib.Path("."):
        path = _RAW / path
    frozen = _FROZEN / f"chain_{match.group(1)}.parquet"
    if not refresh and frozen.exists():
        return pd.read_parquet(frozen)

    raw = pd.read_parquet(path)
    first, last = raw["fetch_utc"].min(), raw["fetch_utc"].max()
    snapshot_date = first.tz_convert(_NY).normalize().tz_localize(None)
    rate, rate_date = _snapshot_rate(_rate_history(), snapshot_date)
    quote_time = raw.loc[raw["expiry"].isin(_selected_expiries(raw)), "lastTradeDate"].max()
    _freeze_saved_bars(path)
    bars, bars_name = _snapshot_bars(path, snapshot_date)
    spots = _spot_entries(bars, bars_name, quote_time, first)
    chain = clean_chain(raw, rate, spots["spot_bar"]["close"])

    _FROZEN.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(frozen)
    _update_manifest({"chains": {frozen.name: {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "snapshot": path.name,
        "ticker": "SPY",
        "yfinance_version": yfinance.__version__,
        "fetch_utc_first": first.isoformat(),
        "fetch_utc_last": last.isoformat(),
        "spot": float(raw["spot_start"].iloc[0]),
        "rate": rate,
        "rate_date": str(rate_date.date()),
        "expiries": [str(e.date()) for e in chain["expiry"].unique()],
        "raw_rows": len(raw),
        "row_count": len(chain),
        "kept": int((chain["status"] == "kept").sum()),
        **spots,
    }}})
    return chain


def nearest_bar_close(bars, when):
    """The 1-minute bar whose close is struck nearest an instant.

    Yahoo labels a bar by its opening minute: bar t covers [t, t + 1 min) and
    its close is the last trade before t + 1 min. The match is on that end
    time; matching the labels would pick a bar whose close comes up to a
    minute after the instant.

    Parameters
    ----------
    bars : pd.DataFrame
        1-minute bars on a tz-aware DatetimeIndex of bar starts, with a Close
        column in currency units (load_minute_bars).
    when : datetime-like
        The instant, tz-aware (tz-naive values are read as UTC).

    Returns
    -------
    dict
        close : float, the matched bar's close
        bar_start, bar_end : pd.Timestamp, the matched bar's start and end in UTC
        Ties go to the earlier bar.

    Raises
    ------
    ValueError
        If bars is empty.
    """
    if bars.empty:
        raise ValueError("no bars to match")
    bars = bars.sort_index()
    starts = pd.DatetimeIndex(bars.index).tz_convert("UTC")
    ends = starts + _BAR
    when = pd.Timestamp(when)
    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    i = int(np.argmin(np.abs((ends - when).to_numpy())))
    return {"close": float(bars["Close"].iloc[i]), "bar_start": starts[i], "bar_end": ends[i]}


class NoBarsError(ValueError):
    """A 1-minute bar download that returned no bars (Yahoo keeps them for about 30 days)."""


def minute_bars_name(ticker, date):
    """File name of a ticker's frozen 1-minute bars: spy_1m_YYYYMMDD.parquet, vix_1m_YYYYMMDD.parquet.

    Parameters
    ----------
    ticker : str
        Yahoo ticker, for example "SPY" or "^VIX"; the name drops a leading
        ^ and is in lower case.
    date : date-like
        The New York trading day.

    Returns
    -------
    str
    """
    return f"{ticker.lstrip('^').lower()}_1m_{pd.Timestamp(date):%Y%m%d}.parquet"


def load_minute_bars(date, refresh=False, ticker="SPY"):
    """1-minute bars of one ticker on one New York trading day, read from data/frozen/ or downloaded.

    The frozen file is data/frozen/<name>_1m_YYYYMMDD.parquet (minute_bars_name:
    spy_1m_... for SPY, vix_1m_... for ^VIX). It is downloaded from yfinance
    (unadjusted) only when it is missing or refresh is True, and the download
    is recorded under "minute_bars" in data/frozen/manifest.json. The bars are
    kept as Yahoo returns them: the regular session for SPY, and for ^VIX also
    the global trading hours bars from 03:15 New York. Yahoo keeps 1-minute
    bars for about 30 days.

    Parameters
    ----------
    date : date-like
        The New York trading day.
    refresh : bool
        If True, download again and overwrite the frozen file.
    ticker : str, default "SPY"
        Yahoo ticker.

    Returns
    -------
    pd.DataFrame
        One row per bar on a DatetimeIndex of bar starts in UTC, named
        bar_start, with columns Open, High, Low, Close (currency units for
        SPY, index points for ^VIX) and Volume (shares; 0 for an index).

    Raises
    ------
    NoBarsError
        If a download returns no bars (a ValueError).
    ValueError
        If a download holds a bar outside the New York date.
    """
    day = pd.Timestamp(date).normalize()
    path = _FROZEN / minute_bars_name(ticker, day)
    if not refresh and path.exists():
        return pd.read_parquet(path)

    raw = yfinance.Ticker(ticker).history(
        start=f"{day:%Y-%m-%d}", end=f"{day + pd.Timedelta(days=1):%Y-%m-%d}", interval="1m", auto_adjust=False
    )
    if raw.empty:
        raise NoBarsError(f"no {ticker} 1-minute bars for {day.date()}; Yahoo keeps them for about 30 days")
    index = pd.DatetimeIndex(raw.index)
    if not (index.tz_convert(_NY).normalize().tz_localize(None) == day).all():
        raise ValueError(f"the download holds bars outside {day.date()} in New York")
    bars = raw[_BAR_COLUMNS].copy()
    bars.index = pd.DatetimeIndex(index.tz_convert("UTC"), freq=None, name="bar_start")  # as read back from parquet

    _FROZEN.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path)
    _update_manifest({"minute_bars": {path.name: {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "yfinance_version": yfinance.__version__,
        "interval": "1m",
        "auto_adjust": False,
        "date": str(day.date()),
        "first_bar": bars.index[0].isoformat(),
        "last_bar": bars.index[-1].isoformat(),
        "row_count": len(bars),
    }}})
    return bars


def _snapshot_bars(path, day, refresh=False):
    """The 1-minute bars of a raw snapshot and the name of the file they come from.

    The bars saved with the pull (spy_1m_YYYYMMDDTHHMMSSZ.parquet next to the
    snapshot path) when they exist, else their frozen copy under the same
    name in data/frozen/, else load_minute_bars(day, refresh), which reads
    data/frozen/spy_1m_YYYYMMDD.parquet and downloads the day only when that
    file is missing or refresh is True.
    """
    name = path.name.replace("spy_chain_", "spy_1m_", 1)
    for saved in (path.with_name(name), _FROZEN / name):
        if saved.exists():
            return pd.read_parquet(saved), name
    return load_minute_bars(day, refresh=refresh), minute_bars_name("SPY", day)


def _freeze_saved_bars(path):
    """Copy the bars saved with a pull into data/frozen/ under the same name, with a manifest entry.

    Returns the frozen copy's path, or None when the pull saved no bars
    (snapshots collected before 2026-10-01).
    """
    name = path.name.replace("spy_chain_", "spy_1m_", 1)
    saved = path.with_name(name)
    if not saved.exists():
        return None
    bars = pd.read_parquet(saved)
    frozen = _FROZEN / name
    _FROZEN.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(frozen)
    starts = pd.DatetimeIndex(bars.index)
    _update_manifest({"minute_bars": {name: {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": "SPY",
        "source": "saved with the pull",
        "snapshot": path.name,
        "interval": "1m",
        "auto_adjust": False,
        "date": str(starts[0].tz_convert(_NY).date()),
        "first_bar": starts[0].isoformat(),
        "last_bar": starts[-1].isoformat(),
        "row_count": len(bars),
    }}})
    return frozen


def _spot_entries(bars, bars_name, quote_time, fetch_first):
    """The spot_bar (nearest quote_time) and spot_bar_fetch (nearest fetch_first) manifest entries."""

    def _entry(when):
        when = pd.Timestamp(when).tz_convert("UTC")
        bar = nearest_bar_close(bars, when)
        return {
            "close": bar["close"],
            "bar_start": bar["bar_start"].isoformat(),
            "bar_end": bar["bar_end"].isoformat(),
            "target_utc": when.isoformat(),
            "bars": bars_name,
        }

    return {"spot_bar": _entry(quote_time), "spot_bar_fetch": _entry(fetch_first)}


def _snapshot_entry(snapshot):
    """The raw path of a snapshot, its frozen chain's name and that chain's manifest entry.

    A bare file name is looked up in data/raw/. Raises ValueError if the name
    does not match spy_chain_YYYYMMDDTHHMMSSZ.parquet or the chain is not in
    the manifest.
    """
    path = pathlib.Path(snapshot)
    match = _SNAPSHOT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path.name} is not a spy_chain_YYYYMMDDTHHMMSSZ.parquet snapshot")
    if path.parent == pathlib.Path("."):
        path = _RAW / path
    name = f"chain_{match.group(1)}.parquet"
    manifest = json.loads(_MANIFEST.read_text()) if _MANIFEST.exists() else {}
    entry = manifest.get("chains", {}).get(name)
    if entry is None:
        raise ValueError(f"{name} is not in the manifest; build it with load_chain first")
    return path, name, entry


def snapshot_spot_bar(snapshot, refresh=False):
    """Spot of a chain snapshot from SPY 1-minute bars, recorded in its manifest entry.

    Yahoo's option quotes lag the fetch by about 15 minutes, so the quotes are
    matched to the spot at the latest option trade in the cleaned chain (the
    quote time), not at the fetch. Two bar closes are taken with
    nearest_bar_close: spot_bar, nearest the quote time, which Table 1's carry
    and the out-of-the-money selection of clean_chain use, and spot_bar_fetch,
    nearest the snapshot's first fetch (fetch_utc_first in the manifest).
    load_chain records both when it builds the chain; here they are written
    to the chain's entry in data/frozen/manifest.json only when they are
    missing or different, so a plain read leaves the manifest unchanged.

    The bars come from the first source that exists: the bars saved with the
    pull, spy_1m_YYYYMMDDTHHMMSSZ.parquet next to the snapshot (under the
    same UTC timestamp, written by scripts/collect_chain.py from 2026-10-01);
    then their frozen copy under the same name in data/frozen/ (written by
    load_chain); then load_minute_bars, which reads
    data/frozen/spy_1m_YYYYMMDD.parquet and downloads the day only when that
    file is missing.

    Parameters
    ----------
    snapshot : str or pathlib.Path
        Raw snapshot spy_chain_YYYYMMDDTHHMMSSZ.parquet, as for load_chain.
        Its cleaned chain must already be frozen and in the manifest.
    refresh : bool
        Passed to load_minute_bars: if True, download the day's bars again.
        It applies only when the snapshot has no saved bars, because those
        record the bars as they stood at the pull.

    Returns
    -------
    dict
        {"spot_bar": entry, "spot_bar_fetch": entry}. Each entry holds close
        (currency units), bar_start, bar_end and target_utc (ISO 8601 strings
        in UTC; target_utc is the instant matched) and bars (the name of the
        bar file read: the saved spy_1m_YYYYMMDDTHHMMSSZ.parquet or the frozen
        spy_1m_YYYYMMDD.parquet).

    Raises
    ------
    ValueError
        If the file name does not match, or the chain is not in the manifest.
    """
    path, name, entry = _snapshot_entry(snapshot)
    chain = load_chain(snapshot)
    fetch_first = pd.Timestamp(entry["fetch_utc_first"])
    day = fetch_first.tz_convert(_NY).normalize().tz_localize(None)
    bars, bars_name = _snapshot_bars(path, day, refresh=refresh)
    spots = _spot_entries(bars, bars_name, chain["lastTradeDate"].max(), fetch_first)
    if any(entry.get(key) != value for key, value in spots.items()):
        _update_manifest({"chains": {name: {**entry, **spots}}})
    return spots


def _daily_close(ticker, day):
    """The daily close of a ticker on one date from yfinance (unadjusted), in its own units."""
    raw = yfinance.download(
        ticker, start=f"{day:%Y-%m-%d}", end=f"{day + pd.Timedelta(days=1):%Y-%m-%d}", auto_adjust=False,
        progress=False, multi_level_index=False,
    )
    if raw.empty or not (pd.DatetimeIndex(raw.index).normalize() == day).any():
        raise ValueError(f"no {ticker} daily close for {day.date()}")
    return float(raw.loc[pd.DatetimeIndex(raw.index).normalize() == day, "Close"].iloc[0])


def snapshot_vix_bar(snapshot, refresh=False):
    """VIX at the quote time of a chain snapshot, recorded in its manifest entry.

    The quote time is the latest option trade in the cleaned chain, the
    instant that spot_bar is matched to (snapshot_spot_bar). VIX there is the
    close of the ^VIX 1-minute bar whose close falls nearest it
    (nearest_bar_close), from load_minute_bars(day, ticker="^VIX"), which
    reads data/frozen/vix_1m_YYYYMMDD.parquet and downloads the day only when
    that file is missing or refresh is True. When no bars are frozen and
    Yahoo returns none (it keeps them for about 30 days), the fallback is the
    daily ^VIX close of the snapshot day. The result is written to the
    chain's entry in data/frozen/manifest.json as vix_bar, only when it is
    missing or different. While no bars are frozen, a recorded daily-close
    fallback is returned as it is, without a download.

    Parameters
    ----------
    snapshot : str or pathlib.Path
        Raw snapshot spy_chain_YYYYMMDDTHHMMSSZ.parquet, as for load_chain.
        Its cleaned chain must already be frozen and in the manifest.
    refresh : bool
        Passed to load_minute_bars: if True, download the day's bars again.

    Returns
    -------
    dict
        close : the VIX level in index points (VIX/100 is the vol as a decimal)
        source : "1-minute bar" or "daily close"
        target_utc : the quote time, ISO 8601 in UTC
        With a 1-minute bar also bar_start and bar_end (ISO 8601 in UTC) and
        bars (the frozen file name); with the daily close, date (the
        snapshot day).

    Raises
    ------
    ValueError
        If the file name does not match, the chain is not in the manifest, or
        neither the bars nor the daily close can be had.
    """
    _, name, entry = _snapshot_entry(snapshot)
    quote_time = load_chain(snapshot)["lastTradeDate"].max().tz_convert("UTC")
    day = pd.Timestamp(entry["fetch_utc_first"]).tz_convert(_NY).normalize().tz_localize(None)
    bars_name = minute_bars_name("^VIX", day)
    recorded = entry.get("vix_bar")
    if (not refresh and recorded is not None and recorded.get("source") == "daily close"
            and not (_FROZEN / bars_name).exists()):
        return recorded

    try:
        bar = nearest_bar_close(load_minute_bars(day, refresh=refresh, ticker="^VIX"), quote_time)
        vix = {
            "close": bar["close"],
            "source": "1-minute bar",
            "target_utc": quote_time.isoformat(),
            "bar_start": bar["bar_start"].isoformat(),
            "bar_end": bar["bar_end"].isoformat(),
            "bars": bars_name,
        }
    except NoBarsError:
        vix = {
            "close": _daily_close("^VIX", day),
            "source": "daily close",
            "target_utc": quote_time.isoformat(),
            "date": str(day.date()),
        }
    if recorded != vix:
        _update_manifest({"chains": {name: {**entry, "vix_bar": vix}}})
    return vix


def cmt_to_rate(y):
    """Continuously compounded rate from a Treasury constant-maturity yield.

    CMT yields are bond-equivalent yields compounded semiannually, so
    (1 + y/2)^(2T) = exp(r·T) gives r = 2·ln(1 + y/2) at every maturity.

    Parameters
    ----------
    y : float, ndarray or pd.Series
        Yield as a decimal (the FRED DGS value / 100).

    Returns
    -------
    Same type as y
        Rate r as a decimal, continuously compounded.
    """
    return 2 * np.log1p(y / 2)


def interp_rate(T, tenors, rates):
    """Rate at each maturity by linear interpolation in T.

    Parameters
    ----------
    T : float or array_like
        Maturities in years.
    tenors : array_like
        Curve maturities in years, increasing.
    rates : array_like
        Continuously compounded rates at the tenors, as decimals.

    Returns
    -------
    float or ndarray
        r(T), held flat at the end rates below the first tenor and above the
        last one.
    """
    return np.interp(T, np.asarray(tenors, dtype=float), np.asarray(rates, dtype=float))


def treasury_rates(curve, date):
    """Treasury rates for a date from a CMT curve table.

    Takes the latest date on or before *date* on which all six yields are
    published and converts them with cmt_to_rate.

    Parameters
    ----------
    curve : pd.DataFrame
        Yields in percent on a DatetimeIndex of dates, one column per FRED
        series (DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3), as returned by
        load_treasury_curve.
    date : date-like
        The date the curve is for.

    Returns
    -------
    curve_date : pd.Timestamp
        The date used.
    tenors : ndarray
        Maturities in years: 1/12, 1/4, 1/2, 1, 2, 3.
    rates : ndarray
        Continuously compounded rates at those tenors, as decimals.

    Raises
    ------
    ValueError
        If no complete curve lies within 5 calendar days on or before *date*.
    """
    day = pd.Timestamp(date).normalize()
    full = curve.loc[curve.index <= day, list(_UST_TENORS)].dropna()
    if full.empty or (day - full.index[-1]).days > _RATE_MAX_AGE_DAYS:
        raise ValueError(f"no complete Treasury curve within {_RATE_MAX_AGE_DAYS} days on or before {day.date()}")
    tenors = np.array(list(_UST_TENORS.values()))
    return full.index[-1], tenors, cmt_to_rate(full.iloc[-1].to_numpy(dtype=float) / 100)


def _fred_series(series, start, end):
    """One FRED series from *start* to *end* as floats in percent on a DatetimeIndex (network)."""
    url = f"{_FRED_CSV}?id={series}&cosd={start:%Y-%m-%d}&coed={end:%Y-%m-%d}"
    frame = pd.read_csv(url, na_values=".")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0]), name="date")
    return pd.Series(frame[series].to_numpy(dtype=float), index=dates, name=series)


def load_treasury_curve(date, refresh=False):
    """Treasury constant-maturity yields around a date, read from data/frozen/ or downloaded.

    The frozen file is data/frozen/ust_curve_YYYYMMDD.csv. It is downloaded
    from FRED (no API key) only when it is missing or refresh is True, over
    the 14 calendar days up to *date*, and recorded under "curves" in
    data/frozen/manifest.json with the curve date treasury_rates selects.

    Parameters
    ----------
    date : date-like
        The date the curve is for (the chain snapshot date).
    refresh : bool
        If True, download again and overwrite the frozen file.

    Returns
    -------
    pd.DataFrame
        Yields in percent as published, one row per date (DatetimeIndex named
        date) and one column per series: DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2,
        DGS3. Unpublished values are NaN.

    Raises
    ------
    ValueError
        If a download holds no complete curve within 5 days on or before
        *date* (nothing is written then).
    """
    day = pd.Timestamp(date).normalize()
    path = _FROZEN / f"ust_curve_{day:%Y%m%d}.csv"
    if refresh or not path.exists():
        start = day - pd.Timedelta(days=_UST_WINDOW_DAYS)
        curve = pd.concat([_fred_series(s, start, day) for s in _UST_TENORS], axis=1).sort_index()
        curve.index.name = "date"
        curve_date, _, _ = treasury_rates(curve, day)

        _FROZEN.mkdir(parents=True, exist_ok=True)
        curve.to_csv(path)
        _update_manifest({"curves": {path.name: {
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "FRED, Treasury constant maturity, percent, bond-equivalent",
            "url": _FRED_CSV,
            "series": list(_UST_TENORS),
            "tenors_years": list(_UST_TENORS.values()),
            "start_date": str(start.date()),
            "end_date": str(day.date()),
            "row_count": len(curve),
            "curve_date": str(curve_date.date()),
        }}})
    return pd.read_csv(path, index_col="date", parse_dates=["date"])
