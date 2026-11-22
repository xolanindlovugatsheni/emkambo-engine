"""Acceptance tests from the Emkambo execution strategy. Seeds are fixed, so results are deterministic."""
import time

import numpy as np
import pandas as pd
import pytest

from emkambo.backtest import run_backtest, kupiec_pof
from emkambo.calibrate import calibrate
from emkambo.engine import simulate_horizon
from emkambo.haircut import HaircutSpec, MCSettings, compute_haircut
from emkambo.index import build_index
from emkambo.params import RegimeOUParams
from emkambo.pipeline import _fit
from emkambo.seasonal import fit_seasonal
from emkambo.synthetic import simulate_market
from emkambo.tallies import weighted_var_es, batch_tally

TRUE = RegimeOUParams(kappa=0.01, sigma1=0.012, sigma2=0.035, q12=1 / 150, q21=1 / 30)


@pytest.fixture(scope="module")
def market():
    truth, trades = simulate_market(TRUE, years=6, p_trade=0.5, seed=1)
    idx, _ = build_index(trades, "maize")
    return truth, idx


@pytest.fixture(scope="module")
def fitted(market):
    _, idx = market
    return _fit(idx, 2)


def test_ou_transition_moments():
    """Single regime (no switching, start calm): exact streaming must match closed-form moments."""
    p = RegimeOUParams(kappa=0.02, sigma1=0.015, sigma2=0.016, q12=1e-12, q21=1.0)
    y0, H, n = 0.1, 60.0, 400_000
    y, w, _ = simulate_horizon(p, y0, 0.0, H, n, np.random.default_rng(3))
    mean_true = y0 * np.exp(-p.kappa * H)
    var_true = p.sigma1**2 * (1 - np.exp(-2 * p.kappa * H)) / (2 * p.kappa)
    se_mean = np.sqrt(var_true / n)
    se_var = var_true * np.sqrt(2 / (n - 1))
    assert abs(y.mean() - mean_true) < 3 * se_mean
    assert abs(y.var(ddof=1) - var_true) < 3 * se_var
    assert np.all(w == 1.0)


@pytest.mark.parametrize("bias,tilt", [(4.0, 0.0), (1.0, 0.75), (4.0, 0.75)])
def test_weights_unbiased(bias, tilt):
    y, w, _ = simulate_horizon(TRUE, 0.0, 0.1, 111, 400_000, np.random.default_rng(11), bias, tilt)
    se = w.std(ddof=1) / np.sqrt(len(w))
    assert abs(w.mean() - 1.0) < 3 * se


def _es_batches(bias, tilt, batches=40, n=20_000, alpha=0.99):
    out = []
    t0 = time.perf_counter()
    for b in range(batches):
        y, w, _ = simulate_horizon(TRUE, 0.0, 0.1, 111, n, np.random.default_rng([b, int(bias * 10), int(tilt * 100)]), bias, tilt)
        out.append(weighted_var_es(-y, w, alpha)[1])
    return batch_tally(out), time.perf_counter() - t0


def test_variance_reduction_unbiased_and_pays():
    analog, t_a = _es_batches(1.0, 0.0)
    biased, t_b = _es_batches(4.0, 0.75)
    combined = np.hypot(analog.std_err, biased.std_err)
    assert abs(analog.mean - biased.mean) < 3 * combined
    fom_a = 1 / (analog.rel_err**2 * t_a)
    fom_b = 1 / (biased.rel_err**2 * t_b)
    assert fom_b / fom_a >= 3.0, f"FOM gain only {fom_b / fom_a:.2f}"


def test_weighted_var_es_matches_unweighted():
    rng = np.random.default_rng(5)
    L = rng.standard_normal(200_000)
    var, es = weighted_var_es(L, np.ones_like(L), 0.99)
    assert abs(var - 2.326) < 0.03 and abs(es - 2.665) < 0.03


def test_parameter_recovery(fitted):
    _, cal, _ = fitted
    z = (cal.theta - TRUE.to_theta()) / cal.se_theta
    assert np.all(np.abs(z) < 3.0), dict(zip(["kappa", "s1", "ds", "q12", "q21"], np.round(z, 2)))


def test_seed_reproducibility(fitted):
    seasonal, cal, origin = fitted
    mc = MCSettings(batches=8, paths_per_batch=5_000, seed=99)
    a = compute_haircut(cal, seasonal, origin, HaircutSpec(tenor_days=60), mc)
    b = compute_haircut(cal, seasonal, origin, HaircutSpec(tenor_days=60), mc)
    for k in ("haircut", "es_point", "es_predictive"):
        assert a[k] == b[k]


def test_tally_quality(fitted):
    seasonal, cal, origin = fitted
    r = compute_haircut(cal, seasonal, origin, HaircutSpec(tenor_days=90),
                        MCSettings(batches=40, paths_per_batch=20_000, param_uncertainty=False))
    assert r["es_tally"]["rel_err"] < 0.05
    assert r["es_tally"]["passes_checks"]


@pytest.mark.slow
def test_kupiec_on_synthetic():
    _, trades = simulate_market(TRUE, years=12, p_trade=0.5, seed=21)
    idx, _ = build_index(trades, "maize")
    spec = HaircutSpec(alpha=0.95, tenor_days=30, liquidation_lag_days=10, seasonal_mode="full")
    res = run_backtest(idx, spec, min_train_days=730, recal_every=6, paths=20_000)
    assert res["kupiec"]["n"] >= 40
    assert res["kupiec"]["p_value"] > 0.05, res["kupiec"]


def test_kupiec_statistic_known_value():
    exc = np.array([1] * 5 + [0] * 95, bool)
    assert kupiec_pof(exc, 0.95)["lr"] == pytest.approx(0.0, abs=1e-12)
