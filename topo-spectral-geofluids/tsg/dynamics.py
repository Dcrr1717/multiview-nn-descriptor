"""
Descriptor-independent dynamical diagnostics used to *label* flow regimes.

Labels must not be derived from any of the spatial descriptors under test,
so they come from the time evolution of each trajectory:

* relative distance to the laminar / conductive state,
* largest Lyapunov exponent from a twin (perturbed) trajectory with periodic
  renormalisation (Benettin et al. 1980),
* temporal coefficient of variation of a global observable (dissipation for
  Kolmogorov flow, Nusselt number for Rayleigh-Benard).
"""
from __future__ import annotations

import numpy as np


def _norm(fields):
    return np.sqrt(sum(float(np.sum(np.abs(f) ** 2)) for f in fields))


def lyapunov_twin(solver, state, T=100.0, tau=1.0, eps=1e-7, seed=0):
    """Largest Lyapunov exponent along `state` (tuple of spectral fields).

    Returns (lambda_mean, lambda_sem, final_state). The SEM is computed from
    block means over 10 blocks, which is conservative for correlated stretches.
    """
    rng = np.random.default_rng(seed)
    pert = []
    for f in state:
        p = rng.standard_normal(f.shape) + 1j * rng.standard_normal(f.shape)
        pert.append(p)
    scale = eps * _norm(state) / _norm(pert)
    twin = tuple(f + scale * p for f, p in zip(state, pert))
    d0 = _norm([a - b for a, b in zip(state, twin)])
    logs = []
    n = int(round(T / tau))
    for _ in range(n):
        state = _as_tuple(solver.advance(_unwrap(state), tau))
        twin = _as_tuple(solver.advance(_unwrap(twin), tau))
        diff = [b - a for a, b in zip(state, twin)]
        d1 = _norm(diff)
        logs.append(np.log(d1 / d0 + 1e-300))
        twin = tuple(a + (d0 / d1) * df for a, df in zip(state, diff))
    logs = np.array(logs) / tau
    blocks = np.array_split(logs, 10)
    bm = np.array([b.mean() for b in blocks])
    return float(logs.mean()), float(bm.std(ddof=1) / np.sqrt(len(bm))), state


def _as_tuple(s):
    return s if isinstance(s, tuple) else (s,)


def _unwrap(s):
    return s[0] if len(s) == 1 else s
