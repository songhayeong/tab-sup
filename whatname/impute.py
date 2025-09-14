import numpy as np
import pandas as pd
from typing import Dict
from .transition import build_Q_from_MR
from .scheduler import CosineSchedule


def impute_column(x_col: pd.Series, y: pd.Series,
                  M: np.ndarray, R: np.ndarray, vals: list,
                  steps=50, dt=0.05, schedule=None, y_star=None, temperature=1.0,
                  rng=None) -> pd.Series:
    """
    if y_star is given -> label-aware guide : R' = onehot(y_star)P(v|y_star)
    none -> R
    """
    rng = np.random.default_rng() if rng is None else rng
    schedule = schedule or CosineSchedule()
    V = len(vals)
    val_to_idx = {v: i for i, v in enumerate(vals)}
    out = x_col.copy()

    # guide : label-conditioned
    R_use = R.copy()
    if y_star is not None:
        # soft guide: P(v|y*)^t
        row = (R[y_star] ** temperature)
        row = row / row.sum()
        R_use = np.zeros_like(R)
        R_use[y_star] = row

    # only missing loc substitute
    miss_mask = x_col.isna() | (x_col.astype(str) == "nan")
    for i in np.where(miss_mask)[0]:
        # initial state starts at the marginal distribution
        p = np.ones(V) / V
        for t in range(steps):
            sigma = schedule(t, steps)
            Q = build_Q_from_MR(M, R_use, sigma)
            p = p + dt * (Q @ p)
            p = np.maximum(p, 0); p /= p.sum()
        # sample
        v_idx = rng.choice(V, p=p)
        out.iloc[i] = vals[v_idx]
    return out
