import math
import pathlib

import pandas as pd
import pytest
import yfinance

from volsurf.data import irx_to_rate, load_history, load_optionmetrics

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
