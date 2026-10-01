"""Tests for scripts/collect_chain.py: time guard, NTM zero-bid abort logic and the bar spot."""

import importlib.util
import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script module without executing main()
# ---------------------------------------------------------------------------
_SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "collect_chain.py"
_spec = importlib.util.spec_from_file_location("collect_chain", _SCRIPT)
_cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cc)

is_market_hours = _cc.is_market_hours
_ntm_zero_bid_share = _cc._ntm_zero_bid_share
BAND = _cc.BAND

NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ny(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=NY)


# Monday 2026-09-28 is a weekday (verified via calendar).
MON = (2026, 9, 28)
SAT = (2026, 9, 26)
SUN = (2026, 9, 27)


# ---------------------------------------------------------------------------
# Time guard tests
# ---------------------------------------------------------------------------

def test_weekday_in_window():
    assert is_market_hours(_ny(*MON, 10, 0)) is True


def test_weekday_open_boundary():
    assert is_market_hours(_ny(*MON, 9, 30, 0)) is True


def test_weekday_before_open():
    assert is_market_hours(_ny(*MON, 9, 29, 59)) is False


def test_weekday_at_close():
    # 16:00:00 is exclusive
    assert is_market_hours(_ny(*MON, 16, 0, 0)) is False


def test_weekday_after_close():
    assert is_market_hours(_ny(*MON, 17, 0)) is False


def test_saturday():
    assert is_market_hours(_ny(*SAT, 11, 0)) is False


def test_sunday():
    assert is_market_hours(_ny(*SUN, 11, 0)) is False


# ---------------------------------------------------------------------------
# NTM zero-bid share tests
# ---------------------------------------------------------------------------

def _make_df(strikes: list[float], bids: list[float], spot: float) -> pd.DataFrame:
    return pd.DataFrame({"strike": strikes, "bid": bids})


def test_ntm_all_bid_nonzero():
    spot = 500.0
    strikes = [spot * np.exp(k) for k in (-0.04, -0.02, 0.0, 0.02, 0.04)]
    df = _make_df(strikes, [1.0] * 5, spot)
    assert _ntm_zero_bid_share(df, spot) == pytest.approx(0.0)


def test_ntm_all_bid_zero():
    spot = 500.0
    strikes = [spot * np.exp(k) for k in (-0.04, -0.02, 0.0, 0.02, 0.04)]
    df = _make_df(strikes, [0.0] * 5, spot)
    assert _ntm_zero_bid_share(df, spot) == pytest.approx(1.0)


def test_ntm_half_zero():
    spot = 500.0
    # 4 NTM strikes: 2 zero-bid, 2 non-zero
    strikes = [spot * np.exp(k) for k in (-0.04, -0.02, 0.02, 0.04)]
    bids = [0.0, 0.0, 1.5, 2.0]
    df = _make_df(strikes, bids, spot)
    assert _ntm_zero_bid_share(df, spot) == pytest.approx(0.5)


def test_far_otm_excluded():
    spot = 500.0
    # 2 NTM rows: both zero-bid  →  share should be 1.0, not diluted by far OTM
    ntm_strikes = [spot * np.exp(k) for k in (-0.03, 0.03)]
    far_strikes = [spot * np.exp(k) for k in (-0.3, -0.2, -0.15, -0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)]
    strikes = ntm_strikes + far_strikes
    bids = [0.0, 0.0] + [0.0] * 10  # everything zero
    df = _make_df(strikes, bids, spot)
    # share uses only NTM rows (2 out of 12)
    share = _ntm_zero_bid_share(df, spot)
    assert share == pytest.approx(1.0)


def test_empty_ntm_returns_zero():
    spot = 500.0
    # All strikes are far OTM → no NTM rows → share = 0.0
    strikes = [spot * np.exp(k) for k in (-0.3, 0.3)]
    df = _make_df(strikes, [0.0, 0.0], spot)
    assert _ntm_zero_bid_share(df, spot) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 1-minute bar spot
# ---------------------------------------------------------------------------

def _bars(closes: list[float], first: datetime) -> pd.DataFrame:
    """A mocked yfinance 1-minute bar frame, indexed by bar start in New York time."""
    index = pd.date_range(first, periods=len(closes), freq="1min")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 100}, index=index)


def test_latest_bar():
    bars = _bars([766.5, 766.6, 766.57], _ny(*MON, 15, 18))
    close, start = _cc.latest_bar(bars)
    assert close == 766.57
    assert start == pd.Timestamp("2026-09-28 19:20", tz="UTC")  # 15:20 New York (EDT)
    assert str(start.tz) == "UTC"


def test_latest_bar_empty():
    close, start = _cc.latest_bar(_bars([], _ny(*MON, 15, 18)))
    assert np.isnan(close) and start is pd.NaT


class _FakeTicker:
    """Stands in for yf.Ticker: one expiry, a constant quote and a new bar frame per history call."""

    def __init__(self, bar_frames):
        self.fast_info = {"last_price": 700.0}
        self.options = ("2026-10-16",)
        self._bar_frames = list(bar_frames)
        self.history_calls = []

    def option_chain(self, expiry):
        side = pd.DataFrame({"contractSymbol": ["A"], "strike": [700.0], "bid": [1.0], "ask": [1.1]})
        return type("Chain", (), {"calls": side, "puts": side.copy()})()

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        frame = self._bar_frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame


def test_fetch_chain_records_bar_spots(monkeypatch):
    start_bars = _bars([701.0, 701.25], _ny(*MON, 15, 19))
    end_bars = _bars([701.25, 701.5, 701.75], _ny(*MON, 15, 19))
    fake = _FakeTicker([start_bars, end_bars])
    monkeypatch.setattr(_cc.yf, "Ticker", lambda ticker: fake)
    monkeypatch.setattr(_cc.time, "sleep", lambda seconds: None)

    spot_start, spot_end, df, bars = _cc._fetch_chain("SPY")
    assert bars is end_bars  # the frame read at the end of the pull, for _save_bars
    assert spot_start == spot_end == 700.0
    assert (df["spot_start"] == 700.0).all() and (df["spot_end"] == 700.0).all()  # the quote is kept
    assert (df["spot_bar_start"] == 701.25).all() and (df["spot_bar_end"] == 701.75).all()
    assert (df["spot_bar_start_utc"] == pd.Timestamp("2026-09-28 19:20", tz="UTC")).all()
    assert (df["spot_bar_end_utc"] == pd.Timestamp("2026-09-28 19:21", tz="UTC")).all()
    assert len(fake.history_calls) == 2
    assert all(call["period"] == "1d" and call["interval"] == "1m" for call in fake.history_calls)


def test_fetch_chain_survives_a_failed_bar_request(monkeypatch):
    fake = _FakeTicker([RuntimeError("no bars"), _bars([], _ny(*MON, 15, 19))])
    monkeypatch.setattr(_cc.yf, "Ticker", lambda ticker: fake)
    monkeypatch.setattr(_cc.time, "sleep", lambda seconds: None)

    _, _, df, bars = _cc._fetch_chain("SPY")
    assert bars.empty
    assert df["spot_bar_start"].isna().all() and df["spot_bar_end"].isna().all()
    assert df["spot_bar_start_utc"].isna().all() and df["spot_bar_end_utc"].isna().all()
    assert str(df["spot_bar_start_utc"].dtype) == "datetime64[ns, UTC]"


# ---------------------------------------------------------------------------
# The day's 1-minute bars saved next to the chain
# ---------------------------------------------------------------------------

def _yahoo_bars(closes: list[float], first: datetime) -> pd.DataFrame:
    """A mocked yfinance history frame with all its columns, indexed by bar start in New York time."""
    bars = _bars(closes, first)
    for column in ("Adj Close", "Dividends", "Stock Splits", "Capital Gains"):
        bars[column] = 0.0
    return bars


def test_bars_up_to():
    bars = _yahoo_bars([766.6, 766.7, 766.57, 766.5, 766.56], _ny(*MON, 15, 18))  # 19:18 to 19:22 UTC
    kept = _cc.bars_up_to(bars, pd.Timestamp("2026-09-28 19:20:47", tz="UTC"))
    assert kept.columns.tolist() == ["Open", "High", "Low", "Close", "Volume"]
    assert kept.index.name == "bar_start" and str(kept.index.tz) == "UTC" and kept.index.freq is None
    assert kept.index.tolist() == [pd.Timestamp(f"2026-09-28 19:{m}", tz="UTC") for m in (18, 19, 20)]
    assert kept["Close"].tolist() == [766.6, 766.7, 766.57]
    # An instant in another time zone is the same instant.
    same = _cc.bars_up_to(bars, pd.Timestamp("2026-09-28 15:20:47", tz="America/New_York"))
    pd.testing.assert_frame_equal(same, kept)


def test_save_bars_writes_the_bars_up_to_the_fetch(tmp_path):
    bars = _yahoo_bars([766.6, 766.7, 766.57, 766.5], _ny(*MON, 15, 18))
    until = pd.Timestamp("2026-09-28 19:20:47", tz="UTC")
    path = tmp_path / "spy_1m_20260928T192047Z.parquet"
    note = _cc._save_bars(bars, until, path)
    assert note == "spy_1m_20260928T192047Z.parquet (3 bars, last 19:20 UTC)"
    pd.testing.assert_frame_equal(pd.read_parquet(path), _cc.bars_up_to(bars, until))


def test_save_bars_warns_and_writes_nothing_on_failure(tmp_path, capsys):
    until = pd.Timestamp("2026-09-28 19:20:47", tz="UTC")
    path = tmp_path / "spy_1m.parquet"
    cases = [
        (None, "the bar request failed"),
        (_yahoo_bars([766.6], _ny(*MON, 15, 25)), "no bars up to the fetch time"),  # starts after the fetch
        (_yahoo_bars([766.6], _ny(*MON, 15, 18)).drop(columns="Close"), "Close"),  # a malformed frame
    ]
    for bars, reason in cases:
        note = _cc._save_bars(bars, until, path)
        assert note.startswith("not saved") and reason in note
        assert reason in capsys.readouterr().err
        assert not path.exists()


def _run_main(tmp_path, monkeypatch, bar_frames):
    """Run main() against a fake Ticker, writing into tmp_path/data/raw."""
    raw = tmp_path / "data" / "raw"
    monkeypatch.setattr(_cc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_cc, "RAW_DIR", raw)
    monkeypatch.setattr(_cc, "is_market_hours", lambda: True)
    monkeypatch.setattr(_cc.yf, "Ticker", lambda ticker: _FakeTicker(bar_frames))
    monkeypatch.setattr(_cc.time, "sleep", lambda seconds: None)
    _cc.main()
    return raw


def test_main_saves_bars_under_the_chain_timestamp(tmp_path, monkeypatch, capsys):
    end_bars = _yahoo_bars([701.25, 701.5], _ny(*MON, 15, 19))  # before the fetch, so both are kept
    raw = _run_main(tmp_path, monkeypatch, [_bars([701.0], _ny(*MON, 15, 19)), end_bars])
    chains = sorted(raw.glob("spy_chain_*.parquet"))
    bar_files = sorted(raw.glob("spy_1m_*.parquet"))
    assert len(chains) == len(bar_files) == 1
    assert bar_files[0].name == chains[0].name.replace("spy_chain_", "spy_1m_")
    until = pd.read_parquet(chains[0])["fetch_utc"].max()
    pd.testing.assert_frame_equal(pd.read_parquet(bar_files[0]), _cc.bars_up_to(end_bars, until))
    assert f"Bars       : {bar_files[0].name} (2 bars" in capsys.readouterr().out


def test_main_keeps_the_chain_when_the_bar_request_fails(tmp_path, monkeypatch, capsys):
    raw = _run_main(tmp_path, monkeypatch, [_bars([701.0], _ny(*MON, 15, 19)), RuntimeError("no bars")])
    assert len(list(raw.glob("spy_chain_*.parquet"))) == 1
    assert not list(raw.glob("spy_1m_*.parquet"))
    captured = capsys.readouterr()
    assert "1-minute bars not saved: the bar request failed" in captured.err
    assert "Bars       : not saved: the bar request failed" in captured.out
