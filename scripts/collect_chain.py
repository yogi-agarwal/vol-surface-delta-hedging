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


def _bar_spot(t) -> tuple[float, pd.Timestamp]:
    """Spot from the latest 1-minute bar of the day; (NaN, NaT) with a warning on failure.

    A failed bar request never stops the chain pull: Yahoo keeps 1-minute bars
    for about 30 days, so the spot can be recovered afterwards.
    """
    try:
        return latest_bar(t.history(period="1d", interval="1m", auto_adjust=False))
    except Exception as exc:
        print(f"  1-minute bar request failed ({exc}); bar spot left missing", file=sys.stderr)
        return float("nan"), pd.NaT


def _fetch_chain(ticker: str) -> tuple[float, float, pd.DataFrame]:
    """Fetch all listed expiries for *ticker*; return (spot_start, spot_end, df).

    spot_start and spot_end are the quote's last price before and after the
    pull. The latest 1-minute bar is read at the same two moments, and df
    carries its close and start time as spot_bar_start, spot_bar_start_utc,
    spot_bar_end and spot_bar_end_utc, next to spot_start and spot_end.
    """
    t = yf.Ticker(ticker)
    spot_start = float(t.fast_info["last_price"])
    bar_start, bar_start_utc = _bar_spot(t)
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
    bar_end, bar_end_utc = _bar_spot(t)

    df = pd.concat(frames, ignore_index=True)
    df["spot_start"] = spot_start
    df["spot_end"] = spot_end
    df["spot_bar_start"] = bar_start
    df["spot_bar_start_utc"] = pd.Series(bar_start_utc, index=df.index, dtype="datetime64[ns, UTC]")
    df["spot_bar_end"] = bar_end
    df["spot_bar_end_utc"] = pd.Series(bar_end_utc, index=df.index, dtype="datetime64[ns, UTC]")
    return spot_start, spot_end, df


def main() -> None:
    if not is_market_hours():
        print(
            "Outside market hours (weekdays 09:30 to 16:00 America/New_York). Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching {TICKER} option chain...")
    spot_start, spot_end, df = _fetch_chain(TICKER)

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


if __name__ == "__main__":
    main()
