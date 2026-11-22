"""Synthetic truth set: price histories from known parameters, observed on a thin exchange.

The latent path is simulated daily with exact OU steps and a daily regime chain
P(1 day) = exp(Q). Trading days are a random subset of calendar days, which
reproduces the irregular gaps of a thin market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import ou_stream
from .params import RegimeOUParams


def simulate_market(p: RegimeOUParams, years: float = 5.0, start: str = "2021-05-01",
                    p_trade: float = 0.5, level_usd: float = 330.0,
                    seasonal_coef=(0.0, 0.08, -0.05, 0.0, 0.0),
                    trades_per_day=(1, 4), trade_noise: float = 0.003, seed: int = 1):
    """Return (truth, trades).

    truth : daily DataFrame with date, regime, y (deviation), s (seasonal), price_usd
    trades: trade-level DataFrame in the exchange input format
    """
    rng = np.random.default_rng(seed)
    n = int(years * 365.25)
    dates = pd.date_range(start, periods=n, freq="D")
    tau = (dates.dayofyear.values - 1) / 365.25
    c = np.asarray(seasonal_coef, float)
    s = np.log(level_usd) + c[0] + c[1] * np.cos(2 * np.pi * tau) + c[2] * np.sin(2 * np.pi * tau) \
        + c[3] * np.cos(4 * np.pi * tau) + c[4] * np.sin(4 * np.pi * tau)

    lam = p.q12 + p.q21
    pi = p.stationary
    d = np.exp(-lam)
    stay = np.array([pi[0] + pi[1] * d, pi[1] + pi[0] * d])
    reg = np.empty(n, dtype=np.int8)
    y = np.empty(n)
    reg[0] = rng.random() < pi[1]
    y[0] = 0.0
    for i in range(1, n):
        reg[i] = reg[i - 1] if rng.random() < stay[reg[i - 1]] else 1 - reg[i - 1]
        y[i] = ou_stream(np.array([y[i - 1]]), 1.0, p.kappa, p.sigmas[reg[i]], rng)[0][0]
    price = np.exp(s + y)
    truth = pd.DataFrame({"date": dates, "regime": reg, "y": y, "s": s, "price_usd": price})

    traded = rng.random(n) < p_trade
    traded[0] = True
    rows = []
    for i in np.flatnonzero(traded):
        k = rng.integers(trades_per_day[0], trades_per_day[1] + 1)
        for _ in range(k):
            rows.append((dates[i], "maize", float(rng.uniform(30, 300)),
                         float(price[i] * np.exp(trade_noise * rng.standard_normal())), "USD"))
    trades = pd.DataFrame(rows, columns=["date", "commodity", "quantity_t", "price", "currency"])
    return truth, trades
