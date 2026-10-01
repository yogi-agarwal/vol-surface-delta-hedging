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
         'spread', 'open interest', 'in the money'

data/frozen/chain_YYYYMMDD.parquet stores this table, so Table 1 rebuilds
from data/frozen/ alone. Snapshots collected after 2026-09-30 also carry
spot_bar_start, spot_bar_end (closes of the latest SPY 1-minute bar at the
start and end of the pull) and spot_bar_start_utc, spot_bar_end_utc (those
bars' start times).

Minute bars (as returned by load_minute_bars)
---------------------------------------------
Open, High, Low, Close : unadjusted SPY prices of each regular-session
                         1-minute bar
Volume                 : shares traded in the bar
Index                  : DatetimeIndex of bar starts in UTC, named bar_start

data/frozen/spy_1m_YYYYMMDD.parquet stores one day's bars; snapshot_spot_bar
matches them to a chain snapshot.

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


def clean_chain(raw, rate):
    """Select expiries, fit parity forwards and label every contract.

    Steps: T per expiry from its own fetch_utc; expiry selection
    (select_expiries); per expiry, D = exp(-r·T) and F from parity_forward
    with D fixed, over the valid pairs with -0.05 <= ln(K/S) <= 0, where S is
    spot_start; filter_quotes on every contract; then out-of-the-money
    selection with F (puts with K < F and calls with K >= F are kept, the
    others are labelled 'in the money'); k = ln(K/F).

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
        spot, a selection outside 8 to 12 slices, or an expiry with fewer
        than 6 valid pairs in the fit band.
    """
    if raw.duplicated(["expiry", "option_type", "strike"]).any():
        raise ValueError("the snapshot repeats an (expiry, option_type, strike) row")
    if not raw["option_type"].isin(["call", "put"]).all():
        raise ValueError("option_type must be 'call' or 'put'")
    if raw.groupby("expiry")["fetch_utc"].nunique().max() > 1:
        raise ValueError("each expiry needs a single fetch_utc")
    if raw["spot_start"].nunique() != 1:
        raise ValueError("the snapshot needs a single spot_start")
    spot = float(raw["spot_start"].iloc[0])

    listed = raw.groupby("expiry")["fetch_utc"].first()
    selected = listed.index[select_expiries(listed.index, time_to_expiry(listed.index, listed.to_numpy()))]
    chain = raw[raw["expiry"].isin(selected)].copy()
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    chain = chain.sort_values(["expiry", "option_type", "strike"], ignore_index=True)

    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain["T"] = time_to_expiry(chain["expiry"], chain["fetch_utc"])
    chain["rate"] = rate(chain["T"].to_numpy()) if callable(rate) else rate
    chain["D"] = np.exp(-chain["rate"] * chain["T"])
    forwards = {}
    for expiry, rows in chain.groupby("expiry"):
        K, *quotes = _parity_pairs(rows)
        x = np.log(K / spot)
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
    in_the_money = np.where(is_call, chain["strike"] < chain["F"], chain["strike"] >= chain["F"])
    chain["status"] = np.where((status == "kept") & in_the_money, "in the money", status)
    return chain


def chain_summary(chain, spot=None):
    """Table 1 per expiry, computed from a cleaned chain alone.

    Parameters
    ----------
    chain : pd.DataFrame
        Output of clean_chain or load_chain.
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
    rules = ["zero bid", "spread", "open interest", "in the money"]
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
    snapshot's New York date (from load_history), written to the frozen
    file, and recorded under "chains" in data/frozen/manifest.json. Nothing
    is downloaded.

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
    rate, rate_date = _snapshot_rate(load_history()["r"], snapshot_date)
    chain = clean_chain(raw, rate)

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


def load_minute_bars(date, refresh=False):
    """SPY 1-minute bars of one New York trading day, read from data/frozen/ or downloaded.

    The frozen file is data/frozen/spy_1m_YYYYMMDD.parquet. It is downloaded
    from yfinance (regular session, unadjusted) only when it is missing or
    refresh is True, and the download is recorded under "minute_bars" in
    data/frozen/manifest.json. Yahoo keeps 1-minute bars for about 30 days.

    Parameters
    ----------
    date : date-like
        The New York trading day.
    refresh : bool
        If True, download again and overwrite the frozen file.

    Returns
    -------
    pd.DataFrame
        One row per bar on a DatetimeIndex of bar starts in UTC, named
        bar_start, with columns Open, High, Low, Close (currency units) and
        Volume (shares).

    Raises
    ------
    ValueError
        If a download returns no bars or a bar outside the New York date.
    """
    day = pd.Timestamp(date).normalize()
    path = _FROZEN / f"spy_1m_{day:%Y%m%d}.parquet"
    if not refresh and path.exists():
        return pd.read_parquet(path)

    raw = yfinance.Ticker("SPY").history(
        start=f"{day:%Y-%m-%d}", end=f"{day + pd.Timedelta(days=1):%Y-%m-%d}", interval="1m", auto_adjust=False
    )
    if raw.empty:
        raise ValueError(f"no SPY 1-minute bars for {day.date()}; Yahoo keeps them for about 30 days")
    index = pd.DatetimeIndex(raw.index)
    if not (index.tz_convert(_NY).normalize().tz_localize(None) == day).all():
        raise ValueError(f"the download holds bars outside {day.date()} in New York")
    bars = raw[_BAR_COLUMNS].copy()
    bars.index = pd.DatetimeIndex(index.tz_convert("UTC"), freq=None, name="bar_start")  # as read back from parquet

    _FROZEN.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path)
    _update_manifest({"minute_bars": {path.name: {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": "SPY",
        "yfinance_version": yfinance.__version__,
        "interval": "1m",
        "auto_adjust": False,
        "date": str(day.date()),
        "first_bar": bars.index[0].isoformat(),
        "last_bar": bars.index[-1].isoformat(),
        "row_count": len(bars),
    }}})
    return bars


def snapshot_spot_bar(snapshot, refresh=False):
    """Spot of a chain snapshot from SPY 1-minute bars, recorded in its manifest entry.

    Yahoo's option quotes lag the fetch by about 15 minutes, so the quotes are
    matched to the spot at the latest option trade in the cleaned chain (the
    quote time), not at the fetch. Two bar closes are taken with
    nearest_bar_close: spot_bar, nearest the quote time, which Table 1's carry
    uses, and spot_bar_fetch, nearest the snapshot's first fetch
    (fetch_utc_first in the manifest). Both are written to the chain's entry
    in data/frozen/manifest.json only when they are missing or different, so
    a plain read leaves the manifest unchanged.

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
        {"spot_bar": entry, "spot_bar_fetch": entry}. Each entry holds close
        (currency units), bar_start, bar_end and target_utc (ISO 8601 strings
        in UTC; target_utc is the instant matched) and bars (the frozen bar
        file's name).

    Raises
    ------
    ValueError
        If the file name does not match, or the chain is not in the manifest.
    """
    match = _SNAPSHOT_NAME.fullmatch(pathlib.Path(snapshot).name)
    if match is None:
        raise ValueError(f"{pathlib.Path(snapshot).name} is not a spy_chain_YYYYMMDDTHHMMSSZ.parquet snapshot")
    name = f"chain_{match.group(1)}.parquet"
    manifest = json.loads(_MANIFEST.read_text()) if _MANIFEST.exists() else {}
    entry = manifest.get("chains", {}).get(name)
    if entry is None:
        raise ValueError(f"{name} is not in the manifest; build it with load_chain first")

    chain = load_chain(snapshot)
    quote_time = pd.Timestamp(chain["lastTradeDate"].max()).tz_convert("UTC")
    fetch_first = pd.Timestamp(entry["fetch_utc_first"]).tz_convert("UTC")
    day = fetch_first.tz_convert(_NY).normalize().tz_localize(None)
    bars = load_minute_bars(day, refresh=refresh)

    def _entry(when):
        bar = nearest_bar_close(bars, when)
        return {
            "close": bar["close"],
            "bar_start": bar["bar_start"].isoformat(),
            "bar_end": bar["bar_end"].isoformat(),
            "target_utc": when.isoformat(),
            "bars": f"spy_1m_{day:%Y%m%d}.parquet",
        }

    spots = {"spot_bar": _entry(quote_time), "spot_bar_fetch": _entry(fetch_first)}
    if any(entry.get(key) != value for key, value in spots.items()):
        _update_manifest({"chains": {name: {**entry, **spots}}})
    return spots


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
