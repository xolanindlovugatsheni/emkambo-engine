"""Collateral haircut: expected shortfall of the loss on a warehouse receipt.

Loss over horizon H (tenor + liquidation lag), per tonne, relative to value today:

    L = 1 - (P_H - c * H) / P_0

with c the storage cost per tonne per day. P_H = P_0 * exp(ds + y_H - y_0), where
ds is the seasonal drift between origin and liquidation and y the stochastic
deviation.

Seasonal drift modes:
  full    - use the fitted seasonal change, up or down
  adverse - use it only when it lowers the price (default: a lender should not
            lend against an expected seasonal rise)
  none    - ignore seasonality in the horizon

Floor price (optional, for maize): with a FloorSpec the lender's sale value is
max(P_H, V_floor) when the floor holds. Three haircuts are then reported:
floor_holds, floor_fails (identical to the no-floor haircut for the same seed)
and blended at the specified p_fail. The headline haircut is the blended one.
See floor.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .calibrate import CalibrationResult
from .engine import simulate_horizon
from .floor import FloorSpec, floor_scenarios
from .params import RegimeOUParams
from .seasonal import SeasonalFit
from .tallies import weighted_var_es, batch_tally, figure_of_merit


@dataclass
class HaircutSpec:
    alpha: float = 0.99
    tenor_days: int = 90
    liquidation_lag_days: int = 21
    storage_usd_per_t_day: float = 0.0
    seasonal_mode: str = "adverse"
    floor: FloorSpec | None = None

    @property
    def horizon(self) -> int:
        return int(self.tenor_days + self.liquidation_lag_days)


@dataclass
class MCSettings:
    batches: int = 40
    paths_per_batch: int = 25_000
    bias: float = 4.0
    tilt: float = 0.75
    param_uncertainty: bool = True
    seed: int = 20261018


@dataclass
class OriginState:
    date: pd.Timestamp
    price_usd: float      # index price per tonne at origin
    y0: float             # deviation at origin
    p_stress0: float      # filtered probability of the stressed regime at origin


def seasonal_drift(seasonal: SeasonalFit | None, origin: pd.Timestamp, horizon: int, mode: str) -> float:
    if seasonal is None or mode == "none":
        return 0.0
    s0, sH = seasonal([origin, origin + pd.Timedelta(days=horizon)])
    ds = float(sH - s0)
    if mode == "full":
        return ds
    if mode == "adverse":
        return min(ds, 0.0)
    raise ValueError(f"unknown seasonal_mode {mode!r}")


def losses(y_h, origin: OriginState, ds: float, spec: HaircutSpec) -> np.ndarray:
    """Market-only loss (no floor)."""
    ratio = np.exp(ds + y_h - origin.y0)
    return 1.0 - ratio + spec.storage_usd_per_t_day * spec.horizon / origin.price_usd


def scenario_losses(y_h, origin: OriginState, ds: float, spec: HaircutSpec, rng) -> dict:
    """Loss arrays by scenario: {'base'} without a floor, or the three floor scenarios.

    Floor draws come from rng after the price simulation, so the price paths and
    the no-floor result are unchanged when a floor is added.
    """
    storage = spec.storage_usd_per_t_day * spec.horizon / origin.price_usd
    ratio = np.exp(ds + y_h - origin.y0)
    if spec.floor is None:
        return {"base": 1.0 - ratio + storage}
    sale = floor_scenarios(ratio, origin.price_usd, spec.floor, rng)
    return {k: 1.0 - v + storage for k, v in sale.items()}


def _primary(scen: dict) -> str:
    return "blended" if "blended" in scen else "base"


def _batch(params: RegimeOUParams, origin, ds, spec, n, rng, bias, tilt):
    """Returns {scenario: (var, es, L)} and the weights."""
    y_h, w, _ = simulate_horizon(params, origin.y0, origin.p_stress0, spec.horizon, n, rng, bias, tilt)
    out = {}
    for k, L in scenario_losses(y_h, origin, ds, spec, rng).items():
        var, es = weighted_var_es(L, w, spec.alpha)
        out[k] = (var, es, L)
    return out, w


def compute_haircut(cal: CalibrationResult, seasonal: SeasonalFit | None, origin: OriginState,
                    spec: HaircutSpec, mc: MCSettings) -> dict:
    """Haircut with Monte Carlo error, FOM and parameter-uncertainty band.

    Two runs share one seed tree:
      1. fixed parameters (point estimate) -> ES, relative error, VOV, FOM
      2. parameters drawn per batch from the calibration covariance ->
         predictive ES (pooled) and the 5-95% band of per-batch ES
    """
    ds = seasonal_drift(seasonal, origin.date, spec.horizon, spec.seasonal_mode)
    root = np.random.SeedSequence(mc.seed)
    point_seeds, unc_seeds, theta_seed = root.spawn(3)

    t0 = time.perf_counter()
    vars_, ess = {}, {}
    for ss in point_seeds.spawn(mc.batches):
        res, _ = _batch(cal.params, origin, ds, spec, mc.paths_per_batch, np.random.default_rng(ss), mc.bias, mc.tilt)
        for k, (v, e, _) in res.items():
            vars_.setdefault(k, []).append(v); ess.setdefault(k, []).append(e)
    elapsed = time.perf_counter() - t0
    key = _primary(ess)
    es_tally = batch_tally(ess[key])
    var_tally = batch_tally(vars_[key])

    out = {
        "alpha": spec.alpha,
        "tenor_days": spec.tenor_days,
        "horizon_days": spec.horizon,
        "seasonal_drift_log": ds,
        "var_point": var_tally.mean,
        "es_point": es_tally.mean,
        "es_tally": es_tally.as_dict(),
        "var_tally": var_tally.as_dict(),
        "fom": figure_of_merit(es_tally.rel_err, elapsed),
        "seconds": elapsed,
        "histories": mc.batches * mc.paths_per_batch,
        "bias": mc.bias,
        "tilt": mc.tilt,
    }

    es_basis = {k: float(np.mean(v)) for k, v in ess.items()}
    if mc.param_uncertainty:
        theta_rng = np.random.default_rng(theta_seed)
        L_all, w_all, es_b = {}, [], []
        for ss in unc_seeds.spawn(mc.batches):
            th = theta_rng.multivariate_normal(cal.theta, cal.cov_theta)
            try:
                p = RegimeOUParams.from_theta(th)
            except ValueError:
                p = cal.params
            res, w = _batch(p, origin, ds, spec, mc.paths_per_batch, np.random.default_rng(ss), mc.bias, mc.tilt)
            for k, (_, _, L) in res.items():
                L_all.setdefault(k, []).append(L)
            w_all.append(w)
            es_b.append(res[key][1])
        w_cat = np.concatenate(w_all)
        es_basis = {k: weighted_var_es(np.concatenate(v), w_cat, spec.alpha)[1] for k, v in L_all.items()}
        out["es_predictive"] = es_basis[key]
        out["es_band_5_95"] = [float(np.percentile(es_b, 5)), float(np.percentile(es_b, 95))]

    out["haircut"] = float(np.clip(es_basis[key], 0.0, 1.0))
    out["advance_rate"] = 1.0 - out["haircut"]
    if spec.floor is not None:
        out["floor"] = spec.floor.as_dict()
        out["haircut_floor_holds"] = float(np.clip(es_basis["floor_holds"], 0.0, 1.0))
        out["haircut_floor_fails"] = float(np.clip(es_basis["floor_fails"], 0.0, 1.0))
        out["floor_dependence"] = out["haircut_floor_fails"] - out["haircut_floor_holds"]
    return out


def haircut_table(cal, seasonal, origin, base_spec: HaircutSpec, tenors, mc: MCSettings) -> pd.DataFrame:
    rows = []
    for tenor in tenors:
        spec = replace(base_spec, tenor_days=int(tenor))
        r = compute_haircut(cal, seasonal, origin, spec, mc)
        rows.append({
            "tenor_days": tenor,
            "horizon_days": r["horizon_days"],
            "alpha": r["alpha"],
            "haircut": r["haircut"],
            "advance_rate": r["advance_rate"],
            "es_point": r["es_point"],
            "es_rel_err": r["es_tally"]["rel_err"],
            "es_vov": r["es_tally"]["vov"],
            "tally_checks_pass": r["es_tally"]["passes_checks"],
            "es_predictive": r.get("es_predictive"),
            "es_band_lo": r.get("es_band_5_95", [None, None])[0],
            "es_band_hi": r.get("es_band_5_95", [None, None])[1],
            "var_point": r["var_point"],
            "fom": r["fom"],
            **({"haircut_floor_holds": r["haircut_floor_holds"],
                "haircut_floor_fails": r["haircut_floor_fails"],
                "floor_dependence": r["floor_dependence"]} if "floor" in r else {}),
        })
    return pd.DataFrame(rows)
