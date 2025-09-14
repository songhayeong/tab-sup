"""
Forward / Reverse Euler step
"""

import numpy as np


def euler_step_dist(p_t: np.ndarray, Q: np.ndarray, dt: float) -> np.ndarray:
    # p_{t+dt} = p_t + dt * W * p_t
    return p_t + dt * (Q @ p_t)


def reverse_generator(Q: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    \bar Q(y, x) = [p_t(y)/p_t(x)] * Q(x, y), with diag fixed so column sum = 0
    Works on square Q (|V| X |V|) given p_t over same support.
    """
    eps = 1e-12
    ratio = (p_t[:, None] / np.maximum(p_t[None, :], eps))
    Qbar = ratio * Q.T  # note: Q(x, y) becomes Q^T(y, x)
    np.fill_diagonal(Qbar, 0.0)
    colsum = Qbar.sum(axis=0)
    np.fill_diagonal(Qbar, -colsum)
    return Qbar


def sample_next(v_idx: int, Q: np.ndarray, dt: float, rng: np.random.Generator) -> int:
    """state v -> one step sampling (delta-t Euler, offdiag prob = dt*Q(offdiag, v))"""
    col = Q[:, v_idx].copy()
    off = col.copy()
    off[v_idx] = 0.0
    p_stay = max(0.0, 1.0 - dt * off.sum())
    probs = off * dt
    probs[v_idx] = p_stay
    probs = probs / probs.sum()
    return rng.choice(len(probs), p=probs)