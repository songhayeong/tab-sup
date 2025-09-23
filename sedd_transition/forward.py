
import numpy as np
from typing import Optional
from transitions import teleport_mix

def dtmc_step(p_row: np.ndarray, A: np.ndarray, sigma: float) -> np.ndarray:
    """One probability-row update: p_{t+1} = (1-sigma)p + sigma (pA)."""
    return (1.0 - sigma) * p_row + sigma * (p_row @ A)

def dtmc_run(p0: np.ndarray, A: np.ndarray, sigmas: np.ndarray, gammas: Optional[np.ndarray]=None) -> np.ndarray:
    """Run DTMC forward over a schedule; returns p_T."""
    p = p0.copy()
    K = A.shape[0]
    for t, sigma in enumerate(sigmas):
        A_t = A
        if gammas is not None:
            A_t = teleport_mix(A, gammas[t])
        p = dtmc_step(p, A_t, float(sigma))
    return p

def sample_step_value(v: int, A: np.ndarray, sigma: float, rng: np.random.Generator) -> int:
    """Single categorical value update: keep with 1-sigma, else sample from row v of A."""
    if rng.random() > sigma:
        return v
    row = A[v]
    return int(rng.choice(len(row), p=row))

def sample_run_values(x: np.ndarray, A: np.ndarray, sigmas: np.ndarray, gammas: Optional[np.ndarray]=None, seed: int=42) -> np.ndarray:
    """Forward-simulate discrete values under DTMC kernel with optional teleport mixing."""
    rng = np.random.default_rng(seed)
    out = x.copy().astype(int)
    for t, sigma in enumerate(sigmas):
        if gammas is not None:
            from .transitions import teleport_mix
            A_t = teleport_mix(A, gammas[t])
        else:
            A_t = A
        for i in range(len(out)):
            out[i] = sample_step_value(out[i], A_t, float(sigma), rng)
    return out
