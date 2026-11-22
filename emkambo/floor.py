"""GMB floor producer price, treated as a put option that can fail.

A lender holding pledged maize can, in principle, deliver it to the Grain
Marketing Board at the floor producer price instead of selling on the market.
The floor's value to the lender is lower than its headline price because:

  * part of the price is paid in ZiG at the interbank rate, which is worth less
    than its USD equivalent at the market rate (zig_market_discount), and the
    ZiG received keeps depreciating until it is paid (zig_depreciation_annual,
    zig_vol_annual);
  * payment arrives after a delay, which costs funding (funding_rate_annual);
  * delivery to a GMB depot costs money (delivery_cost_usd_per_t).

Effective floor value per tonne, in USD at the liquidation date, with t = delay/365:

    V = F * [ (1 - z) + z (1 - g) D ] * exp(-r t) - c_del
    D = exp( -delta t - sigma^2 t / 2 + sigma sqrt(t) Z ),   Z ~ N(0, 1)

so E[D] = exp(-delta t). The floor may also fail (the GMB does not buy, or does
not pay): with probability p_fail the lender gets only the market price.
Lender sale value per tonne = max(market price, V) if the floor holds, else the
market price.

Simplifications, stated in the methodology: ZiG depreciation and floor failure
are drawn independently of the grain price path, and the floor has no quantity
limit. Both are conservative in calm markets and optimistic in a crisis, when a
weak ZiG, GMB funding stress and low grain prices tend to arrive together.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class FloorSpec:
    price_usd_per_t: float               # headline floor producer price F
    zig_share: float = 0.0               # share of F paid in ZiG, z
    zig_market_discount: float = 0.0     # loss from interbank vs market conversion, g
    zig_depreciation_annual: float = 0.0  # expected ZiG depreciation rate, delta
    zig_vol_annual: float = 0.0          # ZiG volatility, sigma
    payment_delay_days: float = 0.0      # days from delivery to payment
    funding_rate_annual: float = 0.0     # lender's USD funding rate, r
    delivery_cost_usd_per_t: float = 0.0  # transport and handling to the depot
    p_fail: float = 0.0                  # probability the floor is not honoured

    def __post_init__(self):
        if not self.price_usd_per_t > 0:
            raise ValueError("price_usd_per_t must be positive")
        for name in ("zig_share", "zig_market_discount", "p_fail"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("zig_depreciation_annual", "zig_vol_annual", "payment_delay_days",
                     "funding_rate_annual", "delivery_cost_usd_per_t"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def t_years(self) -> float:
        return self.payment_delay_days / 365.0

    def expected_value(self) -> float:
        """Closed-form E[V]."""
        t = self.t_years
        zig = self.zig_share * (1 - self.zig_market_discount) * np.exp(-self.zig_depreciation_annual * t)
        return float(self.price_usd_per_t * ((1 - self.zig_share) + zig) * np.exp(-self.funding_rate_annual * t)
                     - self.delivery_cost_usd_per_t)

    def sample_value(self, n: int, rng: np.random.Generator) -> np.ndarray:
        t = self.t_years
        s = self.zig_vol_annual
        if self.zig_share > 0 and s > 0 and t > 0:
            D = np.exp(-self.zig_depreciation_annual * t - 0.5 * s * s * t + s * np.sqrt(t) * rng.standard_normal(n))
        else:
            D = np.full(n, np.exp(-self.zig_depreciation_annual * t))
        val = self.price_usd_per_t * ((1 - self.zig_share) + self.zig_share * (1 - self.zig_market_discount) * D)
        return val * np.exp(-self.funding_rate_annual * t) - self.delivery_cost_usd_per_t

    def as_dict(self) -> dict:
        d = asdict(self)
        d["expected_effective_value_usd_per_t"] = self.expected_value()
        return d


def floor_scenarios(market_ratio: np.ndarray, origin_price: float, floor: FloorSpec,
                    rng: np.random.Generator) -> dict:
    """Sale value relative to origin price, under three floor scenarios.

    market_ratio = P_H / P_0 per path. Uniforms for failure are drawn once, so a
    higher p_fail fails a superset of paths: losses rise pathwise with p_fail.
    """
    n = len(market_ratio)
    v_ratio = floor.sample_value(n, rng) / origin_price
    failed = rng.random(n) < floor.p_fail
    holds = np.maximum(market_ratio, v_ratio)
    return {
        "floor_holds": holds,
        "floor_fails": market_ratio,
        "blended": np.where(failed, market_ratio, holds),
    }
