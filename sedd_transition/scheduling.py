
import numpy as np

def cosine_schedule(T: int) -> np.ndarray:
    """Cosine schedule mapped to 'step intensity' sigma_t in (0,1)."""
    steps = np.arange(T + 1, dtype=float)
    alphas_bar = (np.cos((steps / T + 0.008) / 1.008 * np.pi / 2)) ** 2
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = np.clip(betas, 1e-6, 0.999)
    # Map to [0,1] intensity
    return betas.astype(float)

def linear_schedule(T: int, start: float=1e-4, end: float=1.0) -> np.ndarray:
    return np.linspace(start, end, T, dtype=float)

def gamma_teleport_schedule(T: int, end_gamma: float=0.95) -> np.ndarray:
    """Monotone schedule for teleport mixing coefficient gamma_t."""
    return np.linspace(0.0, end_gamma, T, dtype=float)
