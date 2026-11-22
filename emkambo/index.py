"""Daily price index from trade-level exchange records.

Input columns (CSV): date, commodity, quantity_t, price, currency.
Currency USD is used directly. ZiG (or ZWG) trades are converted with an FX
table (date, zig_per_usd), as-of the trade date; without one they are dropped
and counted. Days without trades are left out, never filled: the gap length
is carried into the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["date", "commodity", "quantity_t", "price", "currency"]


def build_index(trades: pd.DataFrame, commodity: str, fx: pd.DataFrame | None = None,
                mad_k: float = 5.0, window: int = 21):
    missing = [c for c in REQUIRED if c not in trades.columns]
    if missing:
        raise ValueError(f"trade file is missing columns: {missing}")
    report = {"rows_in": int(len(trades))}
    df = trades.loc[trades["commodity"].astype(str).str.strip().str.lower() == commodity.lower(), REQUIRED].copy()
    report["rows_commodity"] = int(len(df))
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_dropped"] = int(before - len(df))
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    df["price_usd"] = np.where(df["currency"] == "USD", df["price"].astype(float), np.nan)

    zig = df["currency"].isin(["ZIG", "ZWG"])
    if zig.any():
        if fx is None:
            report["zig_rows_dropped_no_fx"] = int(zig.sum())
        else:
            f = fx.copy()
            f["date"] = pd.to_datetime(f["date"]).dt.normalize()
            f = f.sort_values("date")[["date", "zig_per_usd"]]
            z = df.loc[zig].sort_values("date")
            z = pd.merge_asof(z, f, on="date", direction="backward")
            df.loc[z.index, "price_usd"] = (z["price"] / z["zig_per_usd"]).values
            report["zig_rows_converted"] = int(z["zig_per_usd"].notna().sum())

    other = ~df["currency"].isin(["USD", "ZIG", "ZWG"])
    report["other_currency_dropped"] = int(other.sum())
    df = df[np.isfinite(df["price_usd"]) & (df["price_usd"] > 0) & (df["quantity_t"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)

    # Robust outlier screen against a rolling median of neighbouring trades.
    lp = np.log(df["price_usd"])
    med = lp.rolling(window, center=True, min_periods=5).median()
    mad = (lp - med).abs().rolling(window, center=True, min_periods=5).median()
    scale = 1.4826 * np.maximum(mad.fillna(np.inf), 0.005)
    flag = ((lp - med).abs() > mad_k * scale) & med.notna()
    report["outliers_dropped"] = int(flag.sum())
    df = df[~flag]

    g = df.assign(pq=df["price_usd"] * df["quantity_t"]).groupby("date")
    idx = pd.DataFrame({
        "price_usd": g["pq"].sum() / g["quantity_t"].sum(),
        "volume_t": g["quantity_t"].sum(),
        "n_trades": g.size(),
    }).reset_index()
    idx["gap_days"] = idx["date"].diff().dt.days
    report["index_days"] = int(len(idx))
    report["median_gap_days"] = float(idx["gap_days"].median()) if len(idx) > 1 else None
    report["max_gap_days"] = float(idx["gap_days"].max()) if len(idx) > 1 else None
    return idx, report
