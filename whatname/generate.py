"""
label condition synthesis
"""
import numpy as np
from .transition import build_Q_from_MR
from .scheduler import CosineSchedule


def sample_values_given_label(M, R, vals, y_star_idx, steps=50, dt=0.05, schedule=None, temperature=1.0, rng=None):
    """
    y* condition sampling for each feature-k
    """
    rng = np.random.default_rng() if rng is None else rng
    schedule = schedule or CosineSchedule()
    V = len(vals)

    # label-aware R
    row = (R[y_star_idx] ** temperature); row = row/row.sum()
    R_use = np.zeros_like(R); R_use[y_star_idx] = row

    # stationary distribution : uniform
    p = np.ones(V)/V
    for t in range(steps):
        sigma = schedule(t, steps)
        Q = build_Q_from_MR(M, R_use, sigma)
        p = p + dt * (Q @ p)
        p = np.maximum(p, 0); p /= p.sum()
    v_idx = rng.choice(V, p=p)
    return vals[v_idx], p