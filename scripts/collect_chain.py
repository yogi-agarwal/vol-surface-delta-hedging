"""Collect SPY option chain snapshots on weekdays during market hours."""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
NY = ZoneInfo("America/New_York")
TICKER = "SPY"
BAND = 0.05
ABORT_THRESHOLD = 0.50
FETCH_PAUSE = 0.5
MAX_RETRIES = 3
BAR_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]  # 1-minute bar columns saved next to the chain
KEEP_COLS = [
    "contractSymbol",
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "volume",
    "openInterest",
    "lastTradeDate",
]


def is_market_hours(now: datetime | None = None) -> bool:
    """Return True if *now* falls within weekday 09:30 to 16:00 America/New_York."""
    if now is None:
        now = datetime.now(tz=NY)
    now_ny = now.astimezone(NY)
    if now_ny.weekday() >= 5:
        return False
    t = now_ny.time()
    from datetime import time as _time
    return _time(9, 30) <= t < _time(16, 0)


def _ntm_zero_bid_share(df: pd.DataFrame, spot: float) -> float:
    """Return fraction of near-the-money contracts (|ln(K/S)| <= BAND) with zero bid."""
    mask = np.abs(np.log(df["strike"] / spot)) <= BAND
    ntm = df.loc[mask]
    if ntm.empty:
        return 0.0
    return float((ntm["bid"] == 0).mean())


def latest_bar(bars: pd.DataFrame) -> tuple[float, pd.Timestamp]:
    """Close and start time of the last bar in a 1-minute bar frame.

    Parameters
    ----------
    bars : pd.DataFrame
        Bars as returned by yfinance Ticker.history: a tz-aware DatetimeIndex
        of bar start times (Yahoo labels each bar by its opening minute) and a
        Close column in currency units.

    Returns
    -------
    close : float
        Close of the last bar (the last trade so far if the bar is still open);
        NaN for an empty frame.
    bar_start_utc : pd.Timestamp
        Start of the last bar in UTC; NaT for an empty frame.
    """
    if bars.empty:
        return float("nan"), pd.NaT
    return float(bars["Close"].iloc[-1]), pd.Timestamp(bars.index[-1]).tz_convert("UTC")


def bars_up_to(bars: pd.DataFrame, until) -> pd.DataFrame:
    """The 1-minute bars that start at or before an instant, in the frozen bar format.

    Parameters
    ----------
    bars : pd.DataFrame
        Bars as returned by yfinance Ticker.history: a tz-aware DatetimeIndex
        of bar start times and at least the columns Open, High, Low, Close
        (currency units) and Volume (shares).
    until : datetime-like
        The instant, tz-aware (the pull's last fetch time).

    Returns
    -------
    pd.DataFrame
        Columns Open, High, Low, Close and Volume on a DatetimeIndex of bar
        starts in UTC named bar_start, keeping the bars that start at or
        before *until*.
    """
    out = bars[BAR_COLUMNS].copy()
    out.index = pd.DatetimeIndex(pd.DatetimeIndex(bars.index).tz_convert("UTC"), freq=None, name="bar_start")
    return out[out.index <= pd.Timestamp(until).tz_convert("UTC")]


def _day_bars(t) -> pd.DataFrame | None:
    """The day's 1-minute bars so far, or None with a warning if the request fails.

    A failed bar request never stops the chain pull.
    """
    try:
        return t.history(period="1d", interval="1m", auto_adjust=False)
    except Exception as exc:
        print(f"  1-minute bar request failed ({exc}); bar spot left missing", file=sys.stderr)
        return None


def _bar_spot(bars: pd.DataFrame | None) -> tuple[float, pd.Timestamp]:
    """latest_bar of a bar frame, or (NaN, NaT) when the request failed."""
    return (float("nan"), pd.NaT) if bars is None else latest_bar(bars)


def _save_bars(bars: pd.DataFrame | None, until, path: Path) -> str:
    """Write the day's bars up to *until* to *path* and describe the result.

    Never raises: when the bars are missing, empty up to *until* or cannot be
    written, it prints a warning and writes nothing, so the pull goes on.
    """
    if bars is None:
        note = "not saved: the bar request failed"
    else:
        try:
            kept = bars_up_to(bars, until)
            if not kept.empty:
                kept.to_parquet(path)
                return f"{path.name} ({len(kept)} bars, last {kept.index[-1]:%H:%M} UTC)"
            note = "not saved: no bars up to the fetch time"
        except Exception as exc:
            note = f"not saved: {exc}"
    print(f"  1-minute bars {note}", file=sys.stderr)
    return note


def _fetch_chain(ticker: str) -> tuple[float, float, pd.DataFrame, pd.DataFrame | None]:
    """Fetch all listed expiries for *ticker*; return (spot_start, spot_end, df, bars).

    spot_start and spot_end are the quote's last price before and after the
    pull. The day's 1-minute bars are read at the same two moments, and df
    carries the latest bar's close and start time as spot_bar_start,
    spot_bar_start_utc, spot_bar_end and spot_bar_end_utc, next to spot_start
    and spot_end. bars is the frame read at the end of the pull (None if that
    request failed), from which main saves the bars up to the fetch time.
    """
    t = yf.Ticker(ticker)
    spot_start = float(t.fast_info["last_price"])
    bar_start, bar_start_utc = _bar_spot(_day_bars(t))
    expiries = t.options

    frames: list[pd.DataFrame] = []
    for i, exp in enumerate(expiries):
        if i > 0:
            time.sleep(FETCH_PAUSE)

        oc = None
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                oc = t.option_chain(exp)
                break
            except Exception as exc:
                last_exc = exc
                delay = 2 ** attempt
                print(
                    f"  expiry {exp}: attempt {attempt + 1} failed ({exc}); "
                    f"retrying in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)

        if oc is None:
            print(
                f"Failed to fetch expiry {exp} after {MAX_RETRIES} attempts: {last_exc}",
                file=sys.stderr,
            )
            raise RuntimeError(f"Unrecoverable fetch error for expiry {exp}") from last_exc

        fetch_utc = datetime.now(timezone.utc)

        for side, label in ((oc.calls, "call"), (oc.puts, "put")):
            available = [c for c in KEEP_COLS if c in side.columns]
            chunk = side[available].copy()
            chunk["expiry"] = exp
            chunk["option_type"] = label
            chunk["fetch_utc"] = fetch_utc
            frames.append(chunk)

    spot_end = float(t.fast_info["last_price"])
    bars = _day_bars(t)
    bar_end, bar_end_utc = _bar_spot(bars)

    df = pd.concat(frames, ignore_index=True)
    df["spot_start"] = spot_start
    df["spot_end"] = spot_end
    df["spot_bar_start"] = bar_start
    df["spot_bar_start_utc"] = pd.Series(bar_start_utc, index=df.index, dtype="datetime64[ns, UTC]")
    df["spot_bar_end"] = bar_end
    df["spot_bar_end_utc"] = pd.Series(bar_end_utc, index=df.index, dtype="datetime64[ns, UTC]")
    return spot_start, spot_end, df, bars


def main() -> None:
    if not is_market_hours():
        print(
            "Outside market hours (weekdays 09:30 to 16:00 America/New_York). Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching {TICKER} option chain...")
    spot_start, spot_end, df, bars = _fetch_chain(TICKER)

    share = _ntm_zero_bid_share(df, spot_start)
    if share > ABORT_THRESHOLD:
        print(
            f"NTM zero-bid share {share:.2%} exceeds threshold {ABORT_THRESHOLD:.0%}. "
            "Quotes look empty: aborting without saving.",
            file=sys.stderr,
        )
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    min_fetch_utc = df["fetch_utc"].min()
    if hasattr(min_fetch_utc, "to_pydatetime"):
        min_fetch_utc = min_fetch_utc.to_pydatetime()
    filename = f"spy_chain_{min_fetch_utc.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    out_path = RAW_DIR / filename
    df.to_parquet(out_path)
    # The day's bars up to the fetch time, under the same UTC timestamp, so the
    # spot at the quote time can be read without Yahoo's intraday history.
    bars_note = _save_bars(bars, df["fetch_utc"].max(), RAW_DIR / filename.replace("spy_chain_", "spy_1m_", 1))

    expiry_count = df["expiry"].nunique()
    contract_count = len(df)
    rel_path = out_path.relative_to(REPO_ROOT)
    print(f"Expiries   : {expiry_count}")
    print(f"Contracts  : {contract_count}")
    print(f"Zero-bid % : {share:.2%} (NTM, |ln(K/S)| <= {BAND})")
    print(f"Quote spot : {spot_start:.4f} at the start, {spot_end:.4f} at the end")
    for label, column in (("start", "spot_bar_start"), ("end", "spot_bar_end")):
        when = df[f"{column}_utc"].iloc[0]
        when = "missing" if pd.isna(when) else f"{when:%H:%M} UTC bar"
        print(f"Bar spot   : {df[column].iloc[0]:.4f} at the {label} ({when})")
    print(f"Saved      : {rel_path}")
    print(f"Bars       : {bars_note}")


if __name__ == "__main__":
    main()
