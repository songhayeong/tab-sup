import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from .config import BuildParams
from .propensity import PropensityModel
from .data import categories, label_classes


def stabilized_ipw(X: pd.DataFrame, xk_name: str, y: pd.Series,
                   prop: PropensityModel, alpha=1.0, clip=(0.0, 50.0)
                   ) -> Tuple[np.ndarray, List[str], List]:
    """M_k[v, y] ~ P(Y|do(X_k=v) (IPW)"""
    vals = categories(X, xk_name)
    y_classes = label_classes(y)
    # marginal proability P(X_k = v)
    p_marg = X[xk_name].astype(str).value_counts(normalize=True)
    # e_k(v|x_-k)
    e_hat = prop.predict_proba(X.drop(columns=[xk_name]))
    class_idx = {v: i for i, v in enumerate(prop.classes_)}
    W = np.zeros(len(X))

    xv = X[xk_name].astype(str).tolist()
    for i, v in enumerate(xv):
        idx = class_idx[v]
        denom = max(e_hat[i, idx], 1e-6)
        numer = float(p_marg.loc[v])
        w = numer / denom
        W[i] = np.clip(w, clip[0], clip[1])

    M = np.zeros((len(vals), len(y_classes)), dtype=float)
    for a, v in enumerate(vals):
        mask_v = (X[xk_name].astype(str) == v).to_numpy()
        wv = W[mask_v]
        denom = wv.sum() + 1e-12
        for b, yj in enumerate(y_classes):
            num = wv[(y[mask_v] == yj).to_numpy()].sum()
            M[a, b] = (num + alpha) / (denom + alpha * len(y_classes))
    return M, y_classes, vals


def conditional_label_to_value(X: pd.DataFrame, xk_name: str, y: pd.Series,
                               alpha=1.0) -> Tuple[np.ndarray, List, List[str]]:
    """R_k[y, v] ~ P(X_k=v | Y=y)"""
    vals = categories(X, xk_name)
    y_classes = label_classes(y)
    R = np.zeros((len(y_classes), len(vals)), dtype=float)
    for b, yj in enumerate(y_classes):
        mask = (y == yj).to_numpy()
        denom = mask.sum() + 1e-12
        for a, v in enumerate(vals):
            num = ((X.loc[mask, xk_name].astype(str) == v)).sum()
            R[b, a] = (num + alpha) / (denom + alpha*len(vals))
    return R, y_classes, vals


def build_Q_from_MR(M: np.ndarray, R: np.ndarray, sigma: float) -> np.ndarray:
    """
    A = M @ R -> shape [|V|, |V|]
    Generator Q : offdiag = sigma * A, diag = -colsum(offdiag)
    (column-stochastic generator: column sums = 0)
    """
    A = M @ R   # [V, V]
    Q = sigma * A
    np.fill_diagonal(Q, 0.0)
    colsum = Q.sum(axis=0)
    np.fill_diagonal(Q, -colsum)
    return Q
