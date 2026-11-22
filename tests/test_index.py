import numpy as np
import pandas as pd
import pytest

from emkambo.index import build_index


def _trades(rows):
    return pd.DataFrame(rows, columns=["date", "commodity", "quantity_t", "price", "currency"])


def test_single_trade_day():
    idx, rep = build_index(_trades([("2024-05-02", "maize", 50, 300.0, "USD")]), "maize")
    assert len(idx) == 1 and idx["price_usd"].iloc[0] == 300.0 and idx["n_trades"].iloc[0] == 1


def test_vwap_and_gaps():
    t = _trades([("2024-05-02", "maize", 100, 300.0, "USD"), ("2024-05-02", "maize", 300, 310.0, "USD"),
                 ("2024-05-06", "maize", 10, 305.0, "USD")])
    idx, _ = build_index(t, "maize")
    assert idx["price_usd"].iloc[0] == pytest.approx(307.5)
    assert idx["gap_days"].iloc[1] == 4


def test_zig_only_day_without_fx_is_dropped():
    t = _trades([("2024-05-02", "maize", 100, 300.0, "USD"), ("2024-05-03", "maize", 100, 8000.0, "ZiG")])
    idx, rep = build_index(t, "maize")
    assert len(idx) == 1 and rep["zig_rows_dropped_no_fx"] == 1


def test_zig_converted_with_fx_asof():
    t = _trades([("2024-05-03", "maize", 100, 8000.0, "ZiG")])
    fx = pd.DataFrame({"date": ["2024-05-01"], "zig_per_usd": [26.0]})
    idx, rep = build_index(t, "maize", fx)
    assert idx["price_usd"].iloc[0] == pytest.approx(8000 / 26)


def test_duplicates_and_other_commodities():
    row = ("2024-05-02", "maize", 100, 300.0, "USD")
    t = _trades([row, row, ("2024-05-02", "soya", 100, 600.0, "USD")])
    idx, rep = build_index(t, "Maize")
    assert rep["duplicates_dropped"] == 1 and len(idx) == 1


def test_outlier_screen():
    rng = np.random.default_rng(0)
    rows = [(f"2024-01-{d:02d}", "maize", 50, 300 * np.exp(0.005 * rng.standard_normal()), "USD") for d in range(1, 29)]
    rows.append(("2024-01-15", "maize", 50, 3000.0, "USD"))  # keying error, x10
    idx, rep = build_index(_trades(rows), "maize")
    assert rep["outliers_dropped"] >= 1
    assert idx["price_usd"].max() < 400


def test_missing_columns():
    with pytest.raises(ValueError):
        build_index(pd.DataFrame({"date": [], "price": []}), "maize")
