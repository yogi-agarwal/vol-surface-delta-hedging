import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest
import yfinance

import volsurf.data
from volsurf.black_scholes import black76_price
from volsurf.data import (
    chain_summary,
    clean_chain,
    filter_quotes,
    irx_to_rate,
    load_chain,
    load_history,
    load_optionmetrics,
    monthly_expiries,
    parity_forward,
    parity_residuals,
    select_expiries,
    time_to_expiry,
)

FROZEN_HISTORY = pathlib.Path(__file__).resolve().parents[1] / "data" / "frozen" / "history.parquet"
OM_STD, OM_SURFACE = "om_spy_std_30d_2021_2025.csv", "om_spy_volsurf_2021_2025.csv"


@pytest.fixture(scope="module")
def history():
    return load_history()


def test_dates_unique_increasing(history):
    assert history.index.is_unique
    assert history.index.is_monotonic_increasing


def test_no_nans(history):
    assert not history.isnull().any().any()


def test_rate_range(history):
    assert ((history["r"] > 0) & (history["r"] < 0.2)).all()


def test_sigma_range(history):
    assert ((history["sigma_i"] > 0.05) & (history["sigma_i"] < 1.0)).all()


def test_no_network_when_frozen(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("network call made when frozen file exists")

    monkeypatch.setattr(yfinance, "download", _raise)
    df = load_history(refresh=False)
    assert not df.empty


def test_irx_to_rate():
    d = 0.0535
    r = irx_to_rate(d)
    print(f"d = {d}: r = {r:.7f}")
    assert r == pytest.approx(0.0546, abs=5e-5)
    # The rate grows the bill price 1 - d·91/360 back to face value over 91/365 years.
    assert math.exp(-r * 91 / 365) == pytest.approx(1 - d * 91 / 360, rel=1e-15)
    assert irx_to_rate(0.0) == 0.0


def test_history_rate_is_converted(history):
    stored = pd.read_parquet(FROZEN_HISTORY)["r"]  # discount yield as frozen
    pd.testing.assert_series_equal(history["r"], irx_to_rate(stored))
    assert (history["r"] > stored).all()


# Synthetic OptionMetrics extracts with the WRDS column names; no licensed data is used.
def _write_std(directory, rows):
    """Write a standardised options file from (date, days, cp_flag, impl_volatility) rows."""
    df = pd.DataFrame(rows, columns=["date", "days", "cp_flag", "impl_volatility"])
    df.insert(0, "secid", 109820)
    df["ticker"] = "SPY"
    df.to_csv(directory / OM_STD, index=False)


def _write_surface(directory, rows):
    """Write a volatility surface file from (date, days, delta, cp_flag, impl_volatility) rows."""
    df = pd.DataFrame(rows, columns=["date", "days", "delta", "cp_flag", "impl_volatility"])
    df.insert(0, "secid", 109820)
    df["ticker"] = "SPY"
    df.to_csv(directory / OM_SURFACE, index=False)


def test_load_optionmetrics_absent(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("network call made by load_optionmetrics")

    monkeypatch.setattr(yfinance, "download", _raise)
    assert load_optionmetrics(tmp_path) is None

    std_only, surface_only = tmp_path / "std_only", tmp_path / "surface_only"
    std_only.mkdir()
    surface_only.mkdir()
    _write_std(std_only, [("2024-01-02", 30, "C", 0.2), ("2024-01-02", 30, "P", 0.22)])
    _write_surface(surface_only, [("2024-01-02", 30, 50, "C", 0.21)])
    assert load_optionmetrics(std_only) is None
    assert load_optionmetrics(surface_only) is None


def test_load_optionmetrics_averages_call_and_put(tmp_path):
    ivs = {"2024-01-02": (0.20, 0.22), "2024-01-03": (0.15, 0.19)}  # (call, put) at days 30
    std_rows = [(d, 30, flag, iv) for d, (c, p) in ivs.items() for flag, iv in (("C", c), ("P", p))]
    std_rows += [("2024-01-02", 60, "C", 0.90), ("2024-01-02", 60, "P", 0.90)]  # other maturity: ignored
    surface = {"2024-01-02": (0.25, 0.17, 0.205), "2024-01-03": (0.23, 0.14, 0.168)}  # 25d put, 25d call, 50d call
    surface_rows = []
    for d, (p25, c25, c50) in surface.items():
        surface_rows += [(d, 30, -25, "P", p25), (d, 30, 25, "C", c25), (d, 30, 50, "C", c50)]
        surface_rows += [(d, 30, -50, "P", 0.99), (d, 30, 30, "C", 0.99)]  # other deltas: ignored
        surface_rows += [(d, 10, -25, "P", 0.99), (d, 60, 25, "C", 0.99), (d, 60, 50, "C", 0.99)]  # other days
    _write_std(tmp_path, std_rows)
    _write_surface(tmp_path, surface_rows)

    om = load_optionmetrics(tmp_path)
    assert isinstance(om.index, pd.DatetimeIndex) and om.index.tz is None and om.index.name == "date"
    assert list(om.index) == [pd.Timestamp(d) for d in ivs]
    assert list(om.columns) == ["sigma_atm", "put_25d", "call_25d", "call_50d"]
    assert om["sigma_atm"].tolist() == [(c + p) / 2 for c, p in ivs.values()]
    assert om[["put_25d", "call_25d", "call_50d"]].to_numpy().tolist() == [list(v) for v in surface.values()]


def test_load_optionmetrics_missing_side_gives_nan(tmp_path):
    _write_std(tmp_path, [("2024-01-02", 30, "C", 0.2), ("2024-01-02", 30, "P", 0.22), ("2024-01-03", 30, "C", 0.15)])
    _write_surface(tmp_path, [("2024-01-02", 30, 50, "C", 0.21)])
    om = load_optionmetrics(tmp_path)
    assert om.loc["2024-01-02", "sigma_atm"] == pytest.approx(0.21, rel=1e-15)
    assert om[["put_25d", "call_25d"]].isna().all().all()
    assert math.isnan(om.loc["2024-01-03", "sigma_atm"])  # no put that day: no one-sided average


def test_load_optionmetrics_rejects_duplicates(tmp_path):
    _write_std(tmp_path, [("2024-01-02", 30, "C", 0.2), ("2024-01-02", 30, "P", 0.22)])
    _write_surface(tmp_path, [("2024-01-02", 30, -25, "P", 0.25), ("2024-01-02", 30, -25, "P", 0.26)])
    with pytest.raises(ValueError, match="surf"):
        load_optionmetrics(tmp_path)
    _write_std(tmp_path, [("2024-01-02", 30, "C", 0.2), ("2024-01-02", 30, "C", 0.21)])
    with pytest.raises(ValueError, match="std"):
        load_optionmetrics(tmp_path)


def test_optionmetrics_real_files(history):
    om = load_optionmetrics()
    if om is None:
        pytest.skip("the licensed OptionMetrics files are not in data/raw/")
    # Integrity checks only: nothing from the licensed files is printed.
    assert om.index.is_unique and om.index.is_monotonic_increasing
    assert om.index.isin(history.index).all()
    assert not om.isna().any().any()
    assert ((om > 0.03) & (om < 1.5)).all().all()


# Synthetic chain snapshots with the collector's columns. Prices are European Black-76 and every
# quote is symmetric about its price, so mids satisfy parity exactly.
FETCH = pd.Timestamp("2026-09-30 19:20:47", tz="UTC")  # 15:20:47 in New York
MONTHLIES = [
    "2026-10-16", "2026-11-20", "2026-12-18", "2027-01-15", "2027-03-19", "2027-06-17",
    "2027-09-17", "2027-12-17", "2028-01-21", "2028-06-16", "2028-12-15", "2029-01-19",
]
# A 2-day weekly, the Thursday daily before a listed third Friday, a weekly, month ends, and an
# October monthly beyond 1 year: none is selected.
NOT_SELECTED = ["2026-10-02", "2026-10-15", "2026-10-23", "2026-11-30", "2026-12-31", "2027-03-31", "2027-10-15"]
SPOT, RATE, CARRY = 500.0, 0.04, 0.012
STRIKES = np.arange(450.0, 550.1, 2.5)


def _smile(k):
    """A skewed smile in log-moneyness, as a decimal vol."""
    return 0.2 - 0.4 * k + 0.5 * k**2


def _synthetic_snapshot(expiries=MONTHLIES + NOT_SELECTED, strikes=STRIKES):
    """A raw snapshot with F = SPOT·exp((RATE - CARRY)·T) and D = exp(-RATE·T) per expiry."""
    frames = []
    for expiry in expiries:
        T = time_to_expiry(expiry, FETCH)
        F, D = SPOT * np.exp((RATE - CARRY) * T), np.exp(-RATE * T)
        k = np.log(strikes / F)
        for option_type, is_call in (("call", True), ("put", False)):
            price = black76_price(F, strikes, T, D, _smile(k), is_call)
            rel = 0.02 + 0.3 * np.abs(k) + (0.01 if is_call else 0.0)  # strike-dependent relative half-spread
            frames.append(pd.DataFrame({
                "contractSymbol": [f"SPY{expiry}{option_type}{K}" for K in strikes],
                "strike": strikes,
                "bid": price * (1 - rel),
                "ask": price * (1 + rel),
                "lastPrice": price,
                "volume": 1.0,
                "openInterest": 100,
                "lastTradeDate": FETCH,
                "expiry": expiry,
                "option_type": option_type,
                "fetch_utc": FETCH,
            }))
    raw = pd.concat(frames, ignore_index=True)
    raw["spot_start"] = raw["spot_end"] = SPOT
    return raw


def test_monthly_expiries():
    listed = [
        "2026-10-15", "2026-10-16", "2026-10-23", "2026-10-30", "2026-11-20", "2026-11-30",
        "2026-12-31", "2027-06-17", "2025-04-17",
    ]
    # 2027-06-17 and 2025-04-17 are Thursdays before an unlisted third Friday (Juneteenth observed,
    # Good Friday); 2026-10-15 is a daily expiry before the listed 2026-10-16.
    assert pd.Timestamp("2027-06-18").day_name() == pd.Timestamp("2025-04-18").day_name() == "Friday"
    expected = [False, True, False, False, True, False, False, True, True]
    assert monthly_expiries(listed).tolist() == expected
    assert monthly_expiries(["2027-06-17", "2027-06-18"]).tolist() == [False, True]


def test_select_expiries():
    listed = MONTHLIES + ["2026-10-23", "2027-10-15", "2026-12-31"]
    T = time_to_expiry(listed, FETCH)
    assert select_expiries(listed, T).tolist() == [True] * 12 + [False] * 3

    T_short = T.copy()
    T_short[0] = 6.9 / 365  # under 7 days to the October monthly
    assert select_expiries(listed, T_short).tolist() == [False] + [True] * 11 + [False] * 3

    with pytest.raises(ValueError, match="7 slices"):
        select_expiries(MONTHLIES[:7], T[:7])
    too_many = MONTHLIES + ["2029-03-16"]  # a further quarterly beyond 1 year
    with pytest.raises(ValueError, match="13 slices"):
        select_expiries(too_many, time_to_expiry(too_many, FETCH))


def test_time_to_expiry_fixed_timestamp():
    # 2026-10-16 16:00 EDT is 20:00 UTC: 16 days and 39 min 13 s after the fetch.
    assert time_to_expiry("2026-10-16", FETCH) == pytest.approx(1_384_753 / 31_536_000, abs=1e-15)
    # 2026-11-20 16:00 EST (daylight saving ends on 1 November) is 21:00 UTC: 51 days and 1 h 39 min 13 s.
    assert time_to_expiry("2026-11-20", FETCH) == pytest.approx((51 * 86_400 + 5_953) / 31_536_000, abs=1e-15)
    both = time_to_expiry(["2026-10-16", "2026-11-20"], FETCH)
    assert both.shape == (2,)
    assert both.tolist() == [time_to_expiry("2026-10-16", FETCH), time_to_expiry("2026-11-20", FETCH)]
    assert time_to_expiry("2026-10-16", FETCH.tz_convert("America/Chicago")) == both[0]  # same instant


def test_parity_forward_recovers_f_and_d():
    # DESIGN section 5 acceptance: a European chain from known F and D gives both back to 1e-8.
    F, T = 512.3, 0.75
    D = math.exp(-0.043 * T)
    K = np.arange(480.0, 545.1, 2.5)
    vol = _smile(np.log(K / F))
    call, put = black76_price(F, K, T, D, vol, True), black76_price(F, K, T, D, vol, False)
    h_call, h_put = 0.01 + 0.002 * np.abs(K - 500), 0.03 + 0.001 * np.abs(K - 530)
    quotes = (call - h_call, call + h_call, put - h_put, put + h_put)

    F_fixed, D_fixed = parity_forward(K, *quotes, D=D)
    F_free, D_free = parity_forward(K, *quotes)
    print(f"fixed D: F error {F_fixed - F:.2e}; free slope: F error {F_free - F:.2e}, D error {D_free - D:.2e}")
    assert abs(F_fixed - F) <= 1e-8 and D_fixed == D
    assert abs(F_free - F) <= 1e-8 and abs(D_free - D) <= 1e-8


def test_parity_forward_weights_and_errors():
    # K + (C_mid - P_mid)/D is 200 at K = 100 and 210 at K = 110; half-spreads of 0.1 and 0.2 on
    # both legs give weights 1/0.02 = 50 and 1/0.08 = 12.5.
    K, h = np.array([100.0, 110.0]), np.array([0.1, 0.2])
    call_mid, put_mid = np.array([101.0, 101.0]), np.array([1.0, 1.0])
    F, D = parity_forward(K, call_mid - h, call_mid + h, put_mid - h, put_mid + h, D=1.0)
    assert F == pytest.approx((50 * 200 + 12.5 * 210) / 62.5, rel=1e-14) and D == 1.0

    with pytest.raises(ValueError, match="at least 2"):
        parity_forward([100.0], [1.0], [1.1], [1.0], [1.1])
    with pytest.raises(ValueError, match="ask > bid"):
        parity_forward(K, [1.0, 1.0], [1.0, 1.1], [1.0, 1.0], [1.1, 1.1])


def test_parity_residuals():
    # C_mid - P_mid = 2.1 and -4.2 against D·(F - K) = 1.98 and -2.97.
    residual, half_spread = parity_residuals(
        [100.0, 105.0], [5.0, 2.0], [5.4, 2.2], [3.0, 6.0], [3.2, 6.6], 102.0, 0.99
    )
    np.testing.assert_allclose(residual, [0.12, -1.23], rtol=0, atol=1e-12)
    np.testing.assert_allclose(half_spread, [0.3, 0.4], rtol=0, atol=1e-12)


def test_clean_chain_synthetic_snapshot():
    chain = clean_chain(_synthetic_snapshot(), RATE)
    assert [str(e.date()) for e in chain["expiry"].unique()] == MONTHLIES

    summary = chain_summary(chain)
    T = time_to_expiry(MONTHLIES, FETCH)
    F, D = SPOT * np.exp((RATE - CARRY) * T), np.exp(-RATE * T)
    np.testing.assert_allclose(summary["F"], F, rtol=0, atol=1e-8)
    np.testing.assert_allclose(summary["D"], D, rtol=0, atol=1e-15)
    np.testing.assert_allclose(summary["F_free"], F, rtol=0, atol=1e-8)
    np.testing.assert_allclose(summary["D_free"], D, rtol=0, atol=1e-8)
    np.testing.assert_allclose(summary["q"], CARRY, rtol=0, atol=1e-10)
    assert (summary["pairs"] == 10).all() and (summary["pairs_upper"] == 10).all()
    assert (summary["within"] == 1).all() and (summary["within_upper"] == 1).all()

    # Every quote passes the filters, so only the in-the-money side is removed.
    itm = [(STRIKES < f).sum() + (STRIKES >= f).sum() for f in F]  # calls below F, puts at or above it
    assert summary["in_the_money"].tolist() == itm
    assert (summary["kept"] == 2 * STRIKES.size - summary["in_the_money"]).all()
    kept = chain[chain["status"] == "kept"]
    assert (kept["strike"] < kept["F"]).eq(kept["option_type"] == "put").all()
    np.testing.assert_allclose(chain["k"], np.log(chain["strike"] / chain["F"]), rtol=0, atol=0)
    np.testing.assert_allclose(chain["mid"], (chain["bid"] + chain["ask"]) / 2, rtol=0, atol=0)


def test_filter_quotes_counts():
    bid = [0.0, 0.0, 1.0, 1.0, 1.0, 0.875, 1.0, 2.0]
    ask = [0.05, 0.0, 1.6, 1.25, 1.1, 1.125, 1.5, 2.1]
    open_interest = [5, 100, 3, 9, 10, 50, 50, 0]
    # Rows 1 and 3 also fail open interest; each counts under its first rule. Row 6 sits exactly
    # on the 0.25 spread limit and row 5 on the open interest limit, so both are kept.
    expected = ["zero bid", "zero bid", "spread", "open interest", "kept", "kept", "spread", "open interest"]
    assert filter_quotes(bid, ask, open_interest).tolist() == expected

    raw = _synthetic_snapshot()
    nov, puts, calls = raw["expiry"] == "2026-11-20", raw["option_type"] == "put", raw["option_type"] == "call"
    zero_bid = nov & puts & (raw["strike"] < 465)  # outside the parity band, so the fit is unchanged
    wide = nov & calls & (raw["strike"] > 535)
    thin = nov & puts & raw["strike"].between(465, 470)
    raw.loc[zero_bid, "bid"] = 0.0
    raw.loc[zero_bid | wide, "openInterest"] = 0  # second failures, never counted
    raw.loc[wide, "ask"] = 2 * raw.loc[wide, "bid"]  # relative spread 2/3
    raw.loc[thin, "openInterest"] = 9
    row = chain_summary(clean_chain(raw, RATE)).loc["2026-11-20"]
    F = SPOT * np.exp((RATE - CARRY) * time_to_expiry("2026-11-20", FETCH))
    itm = (STRIKES < F).sum() + (STRIKES >= F).sum()
    counts = [row[c] for c in ["zero_bid", "spread", "open_interest", "in_the_money", "kept"]]
    assert counts == [6, 6, 3, itm, 2 * STRIKES.size - 15 - itm]
    assert row["kept_puts"] == (STRIKES < F).sum() - 9 and row["kept_calls"] == (STRIKES >= F).sum() - 6


def test_clean_chain_rejects_bad_snapshots():
    coarse = _synthetic_snapshot(strikes=np.arange(400.0, 600.1, 10.0))  # 480, 490, 500 in the fit band
    with pytest.raises(ValueError, match="2026-10-16: 3 valid pairs"):
        clean_chain(coarse, RATE)
    raw = _synthetic_snapshot()
    with pytest.raises(ValueError, match="repeats"):
        clean_chain(pd.concat([raw, raw.iloc[:1]], ignore_index=True), RATE)


def test_load_chain_builds_and_reads_frozen(tmp_path, monkeypatch):
    frozen, raw_dir = tmp_path / "frozen", tmp_path / "raw"
    frozen.mkdir()
    raw_dir.mkdir()
    monkeypatch.setattr(volsurf.data, "_FROZEN", frozen)
    monkeypatch.setattr(volsurf.data, "_MANIFEST", frozen / "manifest.json")
    monkeypatch.setattr(volsurf.data, "_RAW", raw_dir)

    def _raise(*args, **kwargs):
        raise RuntimeError("network call made by load_chain")

    monkeypatch.setattr(yfinance, "download", _raise)
    (frozen / "manifest.json").write_text(json.dumps({"tickers": ["SPY"], "row_count": 3}))
    snapshot = raw_dir / "spy_chain_20260930T192047Z.parquet"
    _synthetic_snapshot().to_parquet(snapshot)

    chain = load_chain(snapshot.name)  # a bare name is looked up in data/raw/
    assert (frozen / "chain_20260930.parquet").exists()
    manifest = json.loads((frozen / "manifest.json").read_text())
    assert manifest["tickers"] == ["SPY"] and manifest["row_count"] == 3  # the history entry is kept
    entry = manifest["chains"]["chain_20260930.parquet"]
    rate = load_history().loc["2026-09-29", "r"]  # the last ^IRX close before the snapshot date
    assert entry["snapshot"] == snapshot.name and entry["rate_date"] == "2026-09-29"
    assert entry["rate"] == rate and (chain["rate"] == rate).all()
    assert entry["expiries"] == MONTHLIES and entry["row_count"] == len(chain)
    assert entry["raw_rows"] == len(MONTHLIES + NOT_SELECTED) * 2 * STRIKES.size
    assert entry["kept"] == (chain["status"] == "kept").sum()

    snapshot.unlink()  # the frozen file is read without the raw snapshot
    pd.testing.assert_frame_equal(load_chain(snapshot), chain)
    with pytest.raises(ValueError, match="spy_chain"):
        load_chain(raw_dir / "chain.parquet")
