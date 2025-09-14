import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression


class PropensityModel:
    """X_-k -> X_k (multiclass) probability e_k(v|x_-k) estimation"""

    def __init__(self, **sk_kwargs):
        self.model = LogisticRegression(multi_class="multinomial", max_iter=1000, **sk_kwargs)
        self.classes_: np.ndarray = None
        self.feature_names_: list[str] = []

    def fit(self, X_minus_k: pd.DataFrame, xk: pd.Series):
        X = pd.get_dummies(X_minus_k, drop_first=False)
        self.feature_names_ = X.columns.tolist()
        y = xk.astype(str).to_numpy()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X_minus_k: pd.DataFrame) -> np.ndarray:
        X = pd.get_dummies(X_minus_k, drop_first=False)
        # align columns
        for c in self.feature_names_:
            if c not in X.columns:
                X[c] = 0
        X = X[self.feature_names_]
        return self.model.predict_proba(X)  # [n, |V_k|]
