import numpy as np
import pandas as pd
from typing import Dict, Tuple

EPS = 1e-12


def empirical_marginal(x: np.ndarray, K: int) -> np.ndarray:
    pi = np.bincount(x, minlength=K).astype(float)
    pi /= max(pi.sum(), EPS)
    return pi


def build_M(df: pd.DataFrame, y_col: str, feat: str, K: int, C: int, alpha_m: float = 1.0) -> np.ndarray:
    """M[v,y] = P(Y=y | X=v) with Laplace smoothing."""
    x = df[feat].astype(int).to_numpy()
    y = df[y_col].astype(int).to_numpy()
    M = np.zeros((K, C), dtype=float)
    for v in range(K):
        mask_v = (x == v)
        denom = mask_v.sum()
        if denom == 0:
            M[v, :] = 1.0 / C
        else:
            for yy in range(C):
                num = (y[mask_v] == yy).sum()
                M[v, yy] = (num + alpha_m) / (denom + alpha_m * C)
    return M


def build_R(df: pd.DataFrame, y_col: str, feat: str, K: int, C: int, alpha_r: float = 1.0) -> np.ndarray:
    """R[y,v] = P(X=v | Y=y) with Laplace smoothing."""
    x = df[feat].astype(int).to_numpy()
    y = df[y_col].astype(int).to_numpy()
    R = np.zeros((C, K), dtype=float)
    for yy in range(C):
        mask_y = (y == yy)
        denom = mask_y.sum()
        if denom == 0:
            R[yy, :] = 1.0 / K
        else:
            for v in range(K):
                num = (x[mask_y] == v).sum()
                R[yy, v] = (num + alpha_r) / (denom + alpha_r * K)
    return R


def build_A(M: np.ndarray, R: np.ndarray) -> np.ndarray:
    """A = M @ R, row-stochastic by construction (defensive renorm)."""
    A = M @ R
    A = np.maximum(A, 0.0)
    A /= (A.sum(axis=1, keepdims=True) + EPS)
    return A


def mh_calibrate(A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Metropolis–Hastings calibration to make pi stationary and kernel reversible."""
    """
    임의의 proposal kernel A를 Metropolis-Hastings로 보정해, 주어진 분포 pi가 stationary가 되도록 보장
    우리가 만든 커널은 샘플오차등과 같은 노이즈로 인해 stationary가 되지 않을 수 있다. 따라서 original margin을 보장해주는 장치
    """
    K = A.shape[0]
    A_prop = A + EPS
    A_prop /= A_prop.sum(axis=1, keepdims=True)
    T = np.zeros_like(A_prop)
    for v in range(K):
        for v2 in range(K):
            if v2 == v:
                continue
            acc = min(1.0, (pi[v2] * A_prop[v2, v]) / (pi[v] * A_prop[v, v2]))
            T[v, v2] = A_prop[v, v2] * acc
        T[v, v] = max(0.0, 1.0 - T[v, :].sum())
    return T


def teleport_mix(A: np.ndarray, gamma: float) -> np.ndarray:
    """(1-gamma)A + gamma U, where U is uniform rows."""
    K = A.shape[0]
    U = np.full((K, K), 1.0 / K, dtype=float)
    return (1.0 - gamma) * A + gamma * U


def stationary_error(pi: np.ndarray, A: np.ndarray, ord: int = 1) -> float:
    piA = pi @ A
    return float(np.linalg.norm(piA - pi, ord=ord))


def build_M_R_A_all(
        df: pd.DataFrame,
        y_col: str,
        feature_bins: Dict[str, int],
        alpha_m: float = 1.0,
        alpha_r: float = 1.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    C = int(df[y_col].nunique())
    Ms, Rs, As, Pis = {}, {}, {}, {}
    for feat, K in feature_bins.items():
        x = df[feat].astype(int).to_numpy()
        pi = empirical_marginal(x, K)
        M = build_M(df, y_col, feat, K, C, alpha_m=alpha_m)
        R = build_R(df, y_col, feat, K, C, alpha_r=alpha_r)
        A = build_A(M, R)
        Ms[feat], Rs[feat], As[feat], Pis[feat] = M, R, A, pi
    return Ms, Rs, As, Pis
