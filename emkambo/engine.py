"""Event-based Monte Carlo engine, organised like a particle-transport code.

One history = one price path from origination to liquidation. Between
"collisions" (regime switches) the deviation is streamed exactly with the OU
transition law, so there is no time-step error. Free-flight times are
exponential with the regime exit rate.

Variance reduction, both with exact weights so weighted tallies stay unbiased:

1. Regime biasing (the analogue of source biasing): switching rates q -> q_tilde
   send more histories into the stressed regime. Regime-path weight

       w_reg = prod_jumps (q_s / q_tilde_s) * exp( - integral (q_S - q_tilde_S) dt ).

2. Shock tilting (the analogue of exponential transform / path stretching):
   each segment's normal shock is drawn from N(-tilt, 1), with weight
   exp(tilt * eps + tilt^2 / 2) per segment.
"""
from __future__ import annotations

import numpy as np

from .params import RegimeOUParams


def ou_stream(y, dt, kappa, sigma, rng, tilt: float = 0.0):
    """Exact OU transition over dt (arrays allowed).

    With tilt > 0 the standard normal shock is drawn from N(-tilt, 1), pushing
    prices down (towards collateral loss). Returns (y_new, log likelihood ratio);
    the log ratio is zero when tilt == 0.
    """
    decay = np.exp(-kappa * dt)
    sd = sigma * np.sqrt(-np.expm1(-2.0 * kappa * dt) / (2.0 * kappa))
    eps = rng.standard_normal(np.shape(y))
    if tilt:
        eps = eps - tilt
        return y * decay + sd * eps, tilt * eps + 0.5 * tilt * tilt
    return y * decay + sd * eps, 0.0


def biased_rates(p: RegimeOUParams, bias: float) -> np.ndarray:
    """Biased exit rates: entry into stress multiplied by `bias`, stress exit divided by sqrt(bias)."""
    if bias <= 0:
        raise ValueError("bias must be positive")
    return np.array([p.q12 * bias, p.q21 / np.sqrt(bias)])


def simulate_horizon(p: RegimeOUParams, y0: float, p_stress0: float, horizon: float, n: int,
                     rng: np.random.Generator, bias: float = 1.0, tilt: float = 0.0):
    """Simulate n histories to `horizon` days.

    Returns
    -------
    y_h : terminal deviation, shape (n,)
    w   : history weights, shape (n,) (all ones when bias == 1 and tilt == 0)
    t_stress : time spent in the stressed regime, shape (n,)
    """
    q = p.exit_rates
    qt = q if bias == 1.0 else biased_rates(p, bias)
    sig = p.sigmas
    y = np.full(n, float(y0))
    s = (rng.random(n) < p_stress0).astype(np.int8)
    t = np.zeros(n)
    logw = np.zeros(n)
    t_stress = np.zeros(n)
    active = np.arange(n)
    while active.size:
        sa = s[active]
        rem = horizon - t[active]
        tau = rng.exponential(1.0 / qt[sa])
        jumped = tau < rem
        dt = np.where(jumped, tau, rem)
        y[active], lr = ou_stream(y[active], dt, p.kappa, sig[sa], rng, tilt)
        logw[active] += lr
        t[active] += dt
        t_stress[active] += dt * sa
        if bias != 1.0:
            logw[active] -= (q[sa] - qt[sa]) * dt
            logw[active[jumped]] += np.log(q[sa[jumped]] / qt[sa[jumped]])
        s[active[jumped]] = 1 - sa[jumped]
        active = active[jumped]
    return y, np.exp(logw), t_stress
