"""Model parameters for the two-regime Ornstein-Uhlenbeck price-deviation model.

Time unit throughout the package is the calendar day. Rates (kappa, q12, q21)
are per day; volatilities are per sqrt(day).

Regime 0 = calm (sigma1), regime 1 = stressed (sigma2 > sigma1). The ordering
is enforced by the parametrisation, which removes label switching.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class RegimeOUParams:
    kappa: float   # mean-reversion speed of the deviation, per day
    sigma1: float  # calm-regime volatility, per sqrt(day)
    sigma2: float  # stressed-regime volatility, per sqrt(day)
    q12: float     # switching rate calm -> stressed, per day
    q21: float     # switching rate stressed -> calm, per day

    def __post_init__(self):
        for name in ("kappa", "sigma1", "sigma2", "q12", "q21"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not self.sigma2 > self.sigma1:
            raise ValueError("sigma2 must exceed sigma1 (stressed regime is the high-volatility one)")

    # Unconstrained coordinates used by the optimiser and for parameter-uncertainty sampling.
    def to_theta(self) -> np.ndarray:
        return np.array([
            np.log(self.kappa),
            np.log(self.sigma1),
            np.log(self.sigma2 - self.sigma1),
            np.log(self.q12),
            np.log(self.q21),
        ])

    @classmethod
    def from_theta(cls, theta) -> "RegimeOUParams":
        t = np.asarray(theta, dtype=float)
        s1 = float(np.exp(t[1]))
        return cls(
            kappa=float(np.exp(t[0])),
            sigma1=s1,
            sigma2=s1 + float(np.exp(t[2])),
            q12=float(np.exp(t[3])),
            q21=float(np.exp(t[4])),
        )

    @property
    def sigmas(self) -> np.ndarray:
        return np.array([self.sigma1, self.sigma2])

    @property
    def exit_rates(self) -> np.ndarray:
        """Rate of leaving each regime (two states, so this is the full generator)."""
        return np.array([self.q12, self.q21])

    @property
    def stationary(self) -> np.ndarray:
        lam = self.q12 + self.q21
        return np.array([self.q21 / lam, self.q12 / lam])

    @property
    def half_life_days(self) -> float:
        return float(np.log(2.0) / self.kappa)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["half_life_days"] = self.half_life_days
        d["mean_calm_spell_days"] = 1.0 / self.q12
        d["mean_stress_spell_days"] = 1.0 / self.q21
        d["stationary_stress_share"] = float(self.stationary[1])
        return d


THETA_NAMES = ["log_kappa", "log_sigma1", "log_sigma2_minus_sigma1", "log_q12", "log_q21"]
