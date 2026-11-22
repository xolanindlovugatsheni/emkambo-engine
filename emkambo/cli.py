"""Command line: emkambo {synth,run,backtest}."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .params import RegimeOUParams
from .pipeline import load_config, run_haircuts, run_backtest_cfg
from .synthetic import simulate_market


def main(argv=None):
    ap = argparse.ArgumentParser(prog="emkambo", description="Emkambo commodity collateral risk engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic trade file from known parameters")
    s.add_argument("--out", required=True)
    s.add_argument("--truth-out")
    s.add_argument("--years", type=float, default=6.0)
    s.add_argument("--p-trade", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=1)

    r = sub.add_parser("run", help="index -> calibrate -> haircut table")
    r.add_argument("--config", required=True)

    b = sub.add_parser("backtest", help="rolling VaR backtest")
    b.add_argument("--config", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "synth":
        p = RegimeOUParams(kappa=0.01, sigma1=0.012, sigma2=0.035, q12=1 / 150, q21=1 / 30)
        truth, trades = simulate_market(p, years=a.years, p_trade=a.p_trade, seed=a.seed)
        trades.to_csv(a.out, index=False)
        if a.truth_out:
            truth.to_csv(a.truth_out, index=False)
        print(json.dumps({"trades": len(trades), "true_params": p.as_dict()}, indent=2))
    elif a.cmd == "run":
        out = run_haircuts(load_config(a.config))
        table = pd.read_csv(out / "haircuts.csv")
        cols = ["tenor_days", "haircut", "advance_rate", "es_rel_err", "es_band_lo", "es_band_hi", "tally_checks_pass"]
        if "haircut_floor_holds" in table.columns:
            cols += ["haircut_floor_holds", "haircut_floor_fails", "floor_dependence"]
        print(table[cols].to_string(index=False))
        print(f"\nrun written to {out}")
    elif a.cmd == "backtest":
        out = run_backtest_cfg(load_config(a.config))
        with open(out / "run.json") as f:
            res = json.load(f)
        print(json.dumps({"kupiec": res["kupiec"], "christoffersen": res["christoffersen"]}, indent=2))
        print(f"\nrun written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
