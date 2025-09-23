"""
 This module produces continuous to discrete through binning.
"""


import numpy as np
import pandas as pd
from typing import Tuple, Optional

def bin_continuous(series: pd.Series, n_bins: int, method: str = "quantile") -> Tuple[pd.Series, np.ndarray]:
    """
    Bin a continuous series into integers [0..n_bins-1].
    Returns (codes, bin_edges_or_quantiles).
    - method="quantile": approximately equal-frequency bins
    - method="uniform": equal-width bins
    """
    s = series.astype(float)
    if n_bins <= 1:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index), np.array([s.min(), s.max()])
    if method == "quantile":
        # handle low unique support
        uniq = s.nunique(dropna=True)
        q = min(max(uniq, 1), n_bins)
        codes = pd.qcut(s.rank(method="first"), q=q, labels=False, duplicates="drop")
        # Retrieve quantile edges for reference
        quantiles = np.linspace(0, 1, q+1)
        edges = np.quantile(s, quantiles)
        return codes.astype(int), edges
    elif method == "uniform":
        codes, edges = pd.cut(s, bins=n_bins, labels=False, retbins=True, include_lowest=True, duplicates="drop")
        return codes.astype(int), edges
    else:
        raise ValueError("method must be 'quantile' or 'uniform'")

def ensure_int_categories(series: pd.Series) -> Tuple[pd.Series, dict]:
    """
    Ensure a categorical/ordinal series is integer-coded [0..K-1].
    Returns (codes, mapping dict from original -> code).
    """
    cat = pd.Categorical(series.astype(str))
    codes = pd.Series(cat.codes, index=series.index)
    mapping = {cat.categories[i]: i for i in range(len(cat.categories))}
    return codes.astype(int), mapping
