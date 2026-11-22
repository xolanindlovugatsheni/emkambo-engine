"""Tests for the GMB floor-price model."""
import numpy as np
import pandas as pd
import pytest

from emkambo.floor import FloorSpec
from emkambo.haircut import HaircutSpec, MCSettings, OriginState, compute_haircut
from emkambo.params import RegimeOUParams
from emkambo.pipeline import floor_from_cfg

TRUE = RegimeOUParams(kappa=0.01, sigma1=0.012, sigma2=0.035, q12=1 / 150, q21=1 / 30)


class _Cal:
    """Minimal stand-in for a CalibrationResult with known parameters."""
    def __init__(self, p):
        self.params = p
        self.theta = p.to_theta()
        self.cov_theta = np.eye(5) * 1e-4


ORIGIN = OriginState(pd.Timestamp("2026-06-10"), price_usd=348.5, y0=0.0, p_stress0=0.1)
MC = MCSettings(batches=10, paths_per_batch=10_000, param_uncertainty=False, seed=5)
REALISTIC = FloorSpec(price_usd_per_t=364.75, zig_share=0.5, zig_market_discount=0.1,
                      zig_depreciation_annual=0.2, zig_vol_annual=0.25, payment_delay_days=45,
                      funding_rate_annual=0.15, delivery_cost_usd_per_t=10.0, p_fail=0.2)


def _run(floor, tenor=90, mc=MC):
    spec = HaircutSpec(tenor_days=tenor, storage_usd_per_t_day=0.1, seasonal_mode="none", floor=floor)
    return compute_haircut(_Cal(TRUE), None, ORIGIN, spec, mc)


def test_expected_value_closed_form():
    rng = np.random.default_rng(1)
    v = REALISTIC.sample_value(400_000, rng)
    se = v.std(ddof=1) / np.sqrt(len(v))
    assert abs(v.mean() - REALISTIC.expected_value()) < 3 * se


def test_floor_fails_equals_no_floor_exactly():
    """Adding a floor must not disturb the price paths: same seed, same no-floor ES."""
    base = _run(None)
    with_floor = _run(REALISTIC)
    assert with_floor["haircut_floor_fails"] == pytest.approx(base["haircut"], abs=1e-12)


def test_floor_ordering():
    r = _run(REALISTIC)
    assert r["haircut_floor_holds"] <= r["haircut"] <= r["haircut_floor_fails"]
    assert r["floor_dependence"] >= 0


def test_haircut_monotone_in_failure_probability():
    hs = [_run(FloorSpec(**{**REALISTIC.__dict__, "p_fail": p}))["haircut"] for p in (0.0, 0.1, 0.5, 1.0)]
    assert all(a <= b + 1e-12 for a, b in zip(hs, hs[1:]))


def test_certain_usd_floor_caps_loss():
    """A USD floor paid at once, never failing, bounds the loss by 1 - (F - storage) / P0."""
    f = FloorSpec(price_usd_per_t=360.0)
    r = _run(f)
    storage = 0.1 * 111
    bound = 1 - (360.0 - storage) / ORIGIN.price_usd
    assert r["haircut_floor_holds"] <= max(bound, 0) + 1e-12


def test_config_parsing():
    assert floor_from_cfg(None) is None
    assert floor_from_cfg({"enabled": False, "price_usd_per_t": 364.75}) is None
    f = floor_from_cfg({"enabled": True, "price_usd_per_t": 364.75, "p_fail": 0.3})
    assert f.price_usd_per_t == 364.75 and f.p_fail == 0.3


@pytest.mark.parametrize("bad", [{"price_usd_per_t": 0}, {"price_usd_per_t": 300, "p_fail": 1.5},
                                 {"price_usd_per_t": 300, "zig_share": -0.1},
                                 {"price_usd_per_t": 300, "payment_delay_days": -1}])
def test_invalid_specs_rejected(bad):
    with pytest.raises(ValueError):
        FloorSpec(**bad)


def test_haircut_table_keeps_floor():
    """Regression: the table must carry the FloorSpec into each tenor, not a dict."""
    from emkambo.haircut import haircut_table
    spec = HaircutSpec(storage_usd_per_t_day=0.1, seasonal_mode="none", floor=REALISTIC)
    mc = MCSettings(batches=4, paths_per_batch=2_000, param_uncertainty=False)
    t = haircut_table(_Cal(TRUE), None, ORIGIN, spec, [30, 90], mc)
    assert {"haircut_floor_holds", "haircut_floor_fails", "floor_dependence"} <= set(t.columns)
    assert len(t) == 2
