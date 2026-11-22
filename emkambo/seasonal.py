"""Deterministic seasonal curve for log prices.

x_t = ln P_t = s(t) + y_t, with s(t) a constant plus K annual Fourier harmonics.
The stochastic model is then fitted to the deviation y_t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

YEAR_DAYS = 365.25


def _design(dates: pd.DatetimeIndex, harmonics: int) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    tau = (d.dayofyear.values - 1) / YEAR_DAYS
    cols = [np.ones(len(d))]
    for k in range(1, harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * tau))
        cols.append(np.sin(2 * np.pi * k * tau))
    return np.column_stack(cols)


@dataclass
class SeasonalFit:
    coef: np.ndarray
    harmonics: int
    residual_sd: float

    def __call__(self, dates) -> np.ndarray:
        dates = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(dates)))
        return _design(dates, self.harmonics) @ self.coef

    def amplitude(self) -> float:
        """Peak-to-trough of the seasonal curve in log points."""
        grid = pd.date_range("2001-01-01", "2001-12-31", freq="D")
        s = self(grid)
        return float(s.max() - s.min())

    def as_dict(self) -> dict:
        return {"coef": self.coef.tolist(), "harmonics": self.harmonics,
                "residual_sd": self.residual_sd, "peak_to_trough_log": self.amplitude()}


def fit_seasonal(dates, log_price, harmonics: int = 2) -> SeasonalFit:
    X = _design(pd.DatetimeIndex(pd.to_datetime(dates)), harmonics)
    y = np.asarray(log_price, dtype=float)
    if len(y) < X.shape[1] * 5:
        raise ValueError("too few observations for a seasonal fit")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return SeasonalFit(coef=coef, harmonics=harmonics, residual_sd=float(resid.std(ddof=X.shape[1])))
