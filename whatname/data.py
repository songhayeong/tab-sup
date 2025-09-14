import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def bin_continuous(df: pd.DataFrame, spec) -> Tuple[pd.DataFrame, Dict[str, List]]:
    """continuous to discrete binning"""
    df = df.copy()
    bins_map = []
    if spec.binning:
        for col, cfg in spec.binning.items():
            if cfg["method"] == "quantile":
                q = cfg.get("bins", 10)
                df[col] = pd.qcut(df[col], q, duplicates="drop")
            elif cfg["method"] == "kmeans":
                from sklearn.cluster import KMeans
                k = cfg.get("bins", 10)
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                vals = df[col].to_numpy().reshape(-1, 1)
                lab = km.fit_predict(vals)
                df[col] = pd.Series(lab, index=df.index).astype(str)
            else:
                raise ValueError("Unsupported binning")
            bins_map[col] = sorted(df[col].unique().tolist())
        return df, bins_map


def categories(df: pd.DataFrame, col: str) -> List:
    vals = df[col].astype(str)
    return sorted(vals.unique().tolist())


def label_classes(y: pd.Series) -> List:
    return sorted(y.unique().tolist())
