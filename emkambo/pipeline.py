"""End-to-end runs driven by a YAML config, with a run manifest so every number traces back."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .backtest import run_backtest
from .calibrate import calibrate
from .floor import FloorSpec
from .haircut import HaircutSpec, MCSettings, OriginState, haircut_table
from .index import build_index
from .seasonal import fit_seasonal


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = Path(path).resolve().parent
    for key in ("trades", "fx"):
        if cfg["data"].get(key):
            p = Path(cfg["data"][key])
            cfg["data"][key] = str(p if p.is_absolute() else base / p)
    out = Path(cfg.get("output_dir", "runs"))
    cfg["output_dir"] = str(out if out.is_absolute() else base / out)
    return cfg


def prepare(cfg: dict):
    d = cfg["data"]
    trades = pd.read_csv(d["trades"])
    fx = pd.read_csv(d["fx"]) if d.get("fx") else None
    idx, report = build_index(trades, d["commodity"], fx)
    return idx, report


def _fit(idx: pd.DataFrame, harmonics: int):
    dates = pd.DatetimeIndex(idx["date"])
    x = np.log(idx["price_usd"].values)
    seasonal = fit_seasonal(dates, x, harmonics)
    y = x - seasonal(dates)
    t = (dates - dates[0]).days.values.astype(float)
    cal = calibrate(t, y)
    origin = OriginState(dates[-1], float(idx["price_usd"].iloc[-1]), float(y[-1]), float(cal.filtered[-1, 1]))
    return seasonal, cal, origin


def _manifest(cfg, kind, extra):
    return {
        "kind": kind,
        "engine_version": __version__,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config": cfg,
        "inputs_sha256": {k: _sha256(cfg["data"][k]) for k in ("trades", "fx") if cfg["data"].get(k)},
        **extra,
    }


def _run_dir(cfg, kind) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(cfg["output_dir"]) / f"{cfg['data']['commodity']}-{kind}-{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def floor_from_cfg(f: dict | None) -> FloorSpec | None:
    """Build a FloorSpec from the haircut.floor config block; None if absent or disabled."""
    if not f or not f.get("enabled", False):
        return None
    return FloorSpec(**{k: v for k, v in f.items() if k != "enabled"})


def run_haircuts(cfg: dict) -> Path:
    idx, report = prepare(cfg)
    m = cfg.get("model", {})
    seasonal, cal, origin = _fit(idx, m.get("harmonics", 2))
    h = cfg["haircut"]
    spec = HaircutSpec(alpha=h["alpha"], liquidation_lag_days=h["liquidation_lag_days"],
                       storage_usd_per_t_day=h.get("storage_usd_per_t_day", 0.0),
                       seasonal_mode=h.get("seasonal_mode", "adverse"),
                       floor=floor_from_cfg(h.get("floor")))
    mcs = cfg.get("monte_carlo", {})
    mc = MCSettings(batches=mcs.get("batches", 40), paths_per_batch=mcs.get("paths_per_batch", 25_000),
                    bias=mcs.get("bias", 4.0), tilt=mcs.get("tilt", 0.75), param_uncertainty=mcs.get("param_uncertainty", True),
                    seed=mcs.get("seed", 20261018))
    table = haircut_table(cal, seasonal, origin, spec, h["tenors_days"], mc)

    out = _run_dir(cfg, "haircut")
    idx.to_csv(out / "index.csv", index=False)
    table.to_csv(out / "haircuts.csv", index=False)
    result = {
        "index_report": report,
        "seasonal": seasonal.as_dict(),
        "calibration": cal.as_dict(),
        "origin": {"date": str(origin.date.date()), "price_usd": origin.price_usd,
                   "deviation_y0": origin.y0, "p_stress0": origin.p_stress0},
        "haircuts": table.to_dict(orient="records"),
    }
    with open(out / "run.json", "w") as f:
        json.dump(_manifest(cfg, "haircut", result), f, indent=2, default=str)
    return out


def run_backtest_cfg(cfg: dict) -> Path:
    idx, report = prepare(cfg)
    h, b = cfg["haircut"], cfg.get("backtest", {})
    spec = HaircutSpec(alpha=b.get("alpha", h["alpha"]), tenor_days=b.get("tenor_days", h["tenors_days"][0]),
                       liquidation_lag_days=h["liquidation_lag_days"],
                       storage_usd_per_t_day=h.get("storage_usd_per_t_day", 0.0),
                       seasonal_mode=h.get("seasonal_mode", "adverse"))
    res = run_backtest(idx, spec, min_train_days=b.get("min_train_days", 730),
                       recal_every=b.get("recal_every", 4), paths=b.get("paths", 40_000),
                       bias=cfg.get("monte_carlo", {}).get("bias", 4.0),
                       tilt=cfg.get("monte_carlo", {}).get("tilt", 0.75),
                       harmonics=cfg.get("model", {}).get("harmonics", 2), seed=b.get("seed", 7))
    out = _run_dir(cfg, "backtest")
    res["records"].to_csv(out / "backtest_records.csv", index=False)
    summary = {k: v for k, v in res.items() if k != "records"}
    with open(out / "run.json", "w") as f:
        json.dump(_manifest(cfg, "backtest", {"index_report": report, "spec": spec.__dict__, **summary}),
                  f, indent=2, default=str)
    return out
