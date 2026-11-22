"""Maximum-likelihood calibration by Hamilton filter on an irregularly sampled series.

Observation k is the deviation y_k at time t_k (days). Over the gap
dt_k = t_k - t_{k-1} the regime transition matrix is P(dt) = exp(Q dt), and the
deviation transition, conditional on regime j over the gap, is the exact OU law

    y_k | y_{k-1}, S=j  ~  N( y_{k-1} e^{-kappa dt},  sigma_j^2 (1 - e^{-2 kappa dt}) / (2 kappa) ).

Approximation (stated in the methodology): the regime is held fixed within each
gap and equal to the regime at the end of the gap. It weakens as gaps lengthen.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .params import RegimeOUParams, THETA_NAMES


def _transition_entries(q12: float, q21: float, dt: np.ndarray):
    lam = q12 + q21
    pi1, pi2 = q21 / lam, q12 / lam
    d = np.exp(-lam * dt)
    p11 = pi1 + pi2 * d
    p22 = pi2 + pi1 * d
    return p11, 1.0 - p11, 1.0 - p22, p22  # p11, p12, p21, p22


def _log_densities(p: RegimeOUParams, dt: np.ndarray, y: np.ndarray):
    k = p.kappa
    decay = np.exp(-k * dt)
    unit_var = -np.expm1(-2.0 * k * dt) / (2.0 * k)
    r = y[1:] - y[:-1] * decay
    out = []
    for s in (p.sigma1, p.sigma2):
        v = s * s * unit_var
        out.append(-0.5 * (np.log(2.0 * np.pi * v) + r * r / v))
    return out


def hamilton_filter(p: RegimeOUParams, dt: np.ndarray, y: np.ndarray, prior=None):
    """Return (log-likelihood, filtered probabilities array of shape (n_obs, 2))."""
    dt = np.asarray(dt, float)
    y = np.asarray(y, float)
    if len(y) != len(dt) + 1:
        raise ValueError("y must have exactly one more element than dt")
    lf1, lf2 = _log_densities(p, dt, y)
    p11, p12, p21, p22 = _transition_entries(p.q12, p.q21, dt)
    xi1, xi2 = (p.stationary if prior is None else np.asarray(prior, float))
    filt = np.empty((len(y), 2))
    filt[0] = (xi1, xi2)
    ll = 0.0
    lf1l, lf2l = lf1.tolist(), lf2.tolist()
    p11l, p12l, p21l, p22l = p11.tolist(), p12.tolist(), p21.tolist(), p22.tolist()
    exp, log = math.exp, math.log
    for i in range(len(dt)):
        a1 = xi1 * p11l[i] + xi2 * p21l[i]
        a2 = xi1 * p12l[i] + xi2 * p22l[i]
        l1, l2 = lf1l[i], lf2l[i]
        mx = l1 if l1 > l2 else l2
        f1 = a1 * exp(l1 - mx)
        f2 = a2 * exp(l2 - mx)
        s = f1 + f2
        if not s > 0.0:
            return -np.inf, filt
        ll += mx + log(s)
        xi1, xi2 = f1 / s, f2 / s
        filt[i + 1, 0] = xi1
        filt[i + 1, 1] = xi2
    return ll, filt


def _neg_ll(theta, dt, y):
    try:
        p = RegimeOUParams.from_theta(theta)
    except ValueError:
        return 1e12
    ll, _ = hamilton_filter(p, dt, y)
    return -ll if np.isfinite(ll) else 1e12


def _numerical_hessian(f, x, h=1e-3):
    n = len(x)
    H = np.empty((n, n))
    fx = f(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        H[i, i] = (f(x + ei) - 2 * fx + f(x - ei)) / h**2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


@dataclass
class CalibrationResult:
    params: RegimeOUParams
    theta: np.ndarray
    cov_theta: np.ndarray
    loglik: float
    n_obs: int
    filtered: np.ndarray = field(repr=False)
    converged: bool = True
    notes: list = field(default_factory=list)

    @property
    def se_theta(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov_theta), 0, None))

    def as_dict(self) -> dict:
        return {
            "params": self.params.as_dict(),
            "theta": dict(zip(THETA_NAMES, self.theta.tolist())),
            "se_theta": dict(zip(THETA_NAMES, self.se_theta.tolist())),
            "cov_theta": self.cov_theta.tolist(),
            "loglik": self.loglik,
            "n_obs": self.n_obs,
            "aic": 2 * len(self.theta) - 2 * self.loglik,
            "stress_prob_last": float(self.filtered[-1, 1]),
            "converged": self.converged,
            "notes": self.notes,
        }


def _starting_points(dt, y):
    inc = np.diff(y) / np.sqrt(dt)
    sd = float(np.std(inc)) or 1e-3
    starts = []
    for kappa in (0.005, 0.02):
        for (f1, f2) in ((0.7, 2.0), (0.9, 3.0)):
            starts.append(RegimeOUParams(kappa, f1 * sd, f2 * sd, 1 / 120, 1 / 30).to_theta())
    return starts


def calibrate(t_days, y, starts=None) -> CalibrationResult:
    """Fit the two-regime OU model to deviations y observed at times t_days."""
    t = np.asarray(t_days, float)
    y = np.asarray(y, float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("observation times must be strictly increasing")
    if len(y) < 60:
        raise ValueError(f"only {len(y)} observations; at least 60 are needed for a two-regime fit")
    dt = np.diff(t)
    best = None
    for x0 in (starts or _starting_points(dt, y)):
        res = minimize(_neg_ll, x0, args=(dt, y), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        res = minimize(_neg_ll, res.x, args=(dt, y), method="BFGS", options={"gtol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    notes = []
    theta = best.x
    H = _numerical_hessian(lambda th: _neg_ll(th, dt, y), theta)
    try:
        cov = np.linalg.inv(H)
        if np.any(np.linalg.eigvalsh((cov + cov.T) / 2) <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        notes.append("Hessian not positive definite; covariance from pseudo-inverse of the PD part")
        w, V = np.linalg.eigh((H + H.T) / 2)
        w = np.where(w > 1e-8, w, np.inf)
        cov = (V / w) @ V.T
        warnings.warn(notes[-1])
    params = RegimeOUParams.from_theta(theta)
    ll, filt = hamilton_filter(params, dt, y)
    return CalibrationResult(params=params, theta=theta, cov_theta=(cov + cov.T) / 2, loglik=ll,
                             n_obs=len(y), filtered=filt, converged=bool(best.success), notes=notes)
