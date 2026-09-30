import math
import pathlib

import pandas as pd
import pytest
import yfinance

from volsurf.data import irx_to_rate, load_history

FROZEN_HISTORY = pathlib.Path(__file__).resolve().parents[1] / "data" / "frozen" / "history.parquet"


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
