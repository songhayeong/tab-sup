import numpy as np


def ess(weights: np.ndarray) -> float:
    w = weights / (weights.sum() + 1e-12)
    return 1.0 / (np.square(w).sum() + 1e-12)