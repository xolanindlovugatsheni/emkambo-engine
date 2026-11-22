"""Basis between the local price and the JSE (SAFEX) price in USD, and cross-hedge ratios.

b_t = ln P_local - ln(F_jse * USD_per_ZAR), modelled as an OU process with exact
transitions over irregular gaps. Mean-reversion speed and residual volatility
tell us whether a JSE hedge actually tracks local prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_ou_irregular(t_days, b) -> dict:
    t = np.asarray(t_days, float); b = np.asarray(b, float)
    dt = np.diff(t)

    def nll(th):
        mu, k, s = th[0], np.exp(th[1]), np.exp(th[2])
        decay = np.exp(-k * dt)
        v = s * s * -np.expm1(-2 * k * dt) / (2 * k)
        r = b[1:] - (mu + (b[:-1] - mu) * decay)
        return 0.5 * np.sum(np.log(2 * np.pi * v) + r * r / v)

    sd = np.std(np.diff(b) / np.sqrt(dt)) or 1e-3
    res = minimize(nll, [b.mean(), np.log(0.02), np.log(sd)], method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
    mu, k, s = res.x[0], float(np.exp(res.x[1])), float(np.exp(res.x[2]))
    return {"mu": float(mu), "kappa": k, "sigma": s, "half_life_days": float(np.log(2) / k),
            "stationary_sd": float(s / np.sqrt(2 * k)), "loglik": float(-res.fun)}


def min_variance_hedge(local: pd.Series, hedge_usd: pd.Series, horizon_days: int) -> dict:
    """Minimum-variance hedge ratio from non-overlapping horizon log returns.

    local, hedge_usd: price series indexed by date. The hedge series is carried
    forward onto local trade dates.
    """
    loc = local.sort_index()
    hed = hedge_usd.sort_index().reindex(loc.index.union(hedge_usd.index)).ffill().reindex(loc.index)
    df = pd.DataFrame({"l": np.log(loc), "h": np.log(hed)}).dropna()
    picks, last = [], None
    for d in df.index:
        if last is None or (d - last).days >= horizon_days:
            picks.append(d); last = d
    r = df.loc[picks].diff().dropna()
    if len(r) < 8:
        raise ValueError("fewer than 8 non-overlapping horizons; hedge ratio not estimable")
    cov = np.cov(r["l"], r["h"], ddof=1)
    h = cov[0, 1] / cov[1, 1]
    eff = cov[0, 1] ** 2 / (cov[0, 0] * cov[1, 1])
    return {"hedge_ratio": float(h), "effectiveness_r2": float(eff), "n_horizons": int(len(r))}
