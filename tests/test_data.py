import pytest
import yfinance

from volsurf.data import load_history


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
