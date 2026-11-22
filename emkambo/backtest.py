"""Rolling backtest of VaR exceedances with non-overlapping horizons.

At each origin the model is fitted only on data up to that date, the VaR of the
horizon loss is computed, and the realised loss to the first trade date at or
after the horizon is recorded. Horizons do not overlap, so exceedances are
independent under a correct model and the Kupiec and Christoffersen tests apply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .calibrate import calibrate, hamilton_filter
from .engine import simulate_horizon
from .haircut import HaircutSpec, OriginState, seasonal_drift, losses
from .seasonal import fit_seasonal
from .tallies import weighted_var_es


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(exceed: np.ndarray, alpha: float) -> dict:
    n = int(len(exceed)); x = int(np.sum(exceed)); p = 1 - alpha
    phat = x / n if n else 0.0
    lr = -2 * ((_xlogy(n - x, 1 - p) + _xlogy(x, p)) - (_xlogy(n - x, 1 - phat) + _xlogy(x, phat)))
    return {"n": n, "exceedances": x, "expected": n * p, "rate": phat, "lr": lr, "p_value": float(chi2.sf(lr, 1))}


def christoffersen_ind(exceed: np.ndarray) -> dict:
    e = np.asarray(exceed, int)
    if len(e) < 3:
        return {"lr": float("nan"), "p_value": float("nan")}
    a, b = e[:-1], e[1:]
    n00 = int(np.sum((a == 0) & (b == 0))); n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0))); n11 = int(np.sum((a == 1) & (b == 1)))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    l1 = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr = -2 * (l0 - l1)
    return {"lr": lr, "p_value": float(chi2.sf(lr, 1)), "n01": n01, "n11": n11}


def run_backtest(index_df: pd.DataFrame, spec: HaircutSpec, min_train_days: int = 730,
                 recal_every: int = 4, paths: int = 40_000, bias: float = 4.0, tilt: float = 0.75,
                 harmonics: int = 2, seed: int = 7) -> dict:
    idx = index_df.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(idx["date"])
    x = np.log(idx["price_usd"].values)
    t = (dates - dates[0]).days.values.astype(float)
    H = spec.horizon
    seeds = np.random.SeedSequence(seed)

    records, i_origin, k = [], int(np.searchsorted(t, min_train_days)), 0
    seasonal = cal = None
    while i_origin < len(t):
        i_liq = int(np.searchsorted(t, t[i_origin] + H))
        if i_liq >= len(t):
            break
        tr = slice(0, i_origin + 1)
        if k % recal_every == 0:
            seasonal = fit_seasonal(dates[tr], x[tr], harmonics)
            y_tr = x[tr] - seasonal(dates[tr])
            cal = calibrate(t[tr], y_tr)
            p_stress = float(cal.filtered[-1, 1])
        else:
            y_tr = x[tr] - seasonal(dates[tr])
            _, filt = hamilton_filter(cal.params, np.diff(t[tr]), y_tr)
            p_stress = float(filt[-1, 1])
        origin = OriginState(dates[i_origin], float(np.exp(x[i_origin])), float(y_tr[-1]), p_stress)
        ds = seasonal_drift(seasonal, origin.date, H, spec.seasonal_mode)
        rng = np.random.default_rng(seeds.spawn(1)[0])
        y_h, w, _ = simulate_horizon(cal.params, origin.y0, p_stress, H, paths, rng, bias, tilt)
        var, es = weighted_var_es(losses(y_h, origin, ds, spec), w, spec.alpha)
        h_act = float(t[i_liq] - t[i_origin])
        realised = 1 - (np.exp(x[i_liq]) - spec.storage_usd_per_t_day * h_act) / origin.price_usd
        records.append({"origin": origin.date.date(), "liquidation": dates[i_liq].date(),
                        "p_stress": p_stress, "var": var, "es": es,
                        "realised_loss": realised, "exceed": bool(realised > var)})
        i_origin, k = i_liq, k + 1

    rec = pd.DataFrame(records)
    exc = rec["exceed"].values if len(rec) else np.array([], bool)
    return {"records": rec,
            "kupiec": kupiec_pof(exc, spec.alpha) if len(exc) else None,
            "christoffersen": christoffersen_ind(exc) if len(exc) else None,
            "mean_es_when_exceeded": float(rec.loc[rec.exceed, "es"].mean()) if exc.any() else None,
            "mean_loss_when_exceeded": float(rec.loc[rec.exceed, "realised_loss"].mean()) if exc.any() else None}
