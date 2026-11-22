"""Tallies with self-reported error, in the manner of a transport code.

Weighted VaR/ES use the unbiased (not self-normalised) estimator: the tail
probability is (1/N) * sum of weights above a level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def weighted_var_es(loss: np.ndarray, w: np.ndarray, alpha: float):
    """VaR and ES at level alpha for weighted samples (Rockafellar-Uryasev)."""
    loss = np.asarray(loss, float)
    w = np.asarray(w, float)
    n = len(loss)
    tail = 1.0 - alpha
    order = np.argsort(-loss)
    ls, ws = loss[order], w[order]
    cum = np.cumsum(ws) / n
    k = int(np.searchsorted(cum, tail, side="right"))
    k = min(k, n - 1)
    var = ls[k]
    es = var + np.sum(ws * np.clip(ls - var, 0, None)) / (n * tail)
    return float(var), float(es)


@dataclass
class BatchTally:
    mean: float
    std_err: float
    rel_err: float
    vov: float
    n_batches: int

    def passes(self, rel_tol=0.05, vov_tol=0.10) -> bool:
        # VOV here is computed over batch estimates, whose normal-theory value is
        # about 2/B; the MCNP per-history threshold of 0.1 is applied above that floor.
        floor = 4.0 / self.n_batches
        return self.rel_err < rel_tol and self.vov < max(vov_tol, floor)

    def as_dict(self):
        return {"mean": self.mean, "std_err": self.std_err, "rel_err": self.rel_err,
                "vov": self.vov, "n_batches": self.n_batches, "passes_checks": self.passes()}


def batch_tally(values) -> BatchTally:
    x = np.asarray(values, float)
    b = len(x)
    m = float(x.mean())
    dev = x - m
    se = float(dev.std(ddof=1) / np.sqrt(b)) if b > 1 else float("nan")
    rel = abs(se / m) if m != 0 else float("inf")
    s2 = float(np.sum(dev**2))
    vov = float(np.sum(dev**4) / s2**2 - 1.0 / b) if s2 > 0 else 0.0
    return BatchTally(mean=m, std_err=se, rel_err=rel, vov=vov, n_batches=b)


def figure_of_merit(rel_err: float, seconds: float) -> float:
    return 1.0 / (rel_err**2 * seconds) if rel_err > 0 and seconds > 0 else float("inf")
