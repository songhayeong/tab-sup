
import numpy as np
import pandas as pd

def mat_to_df(mat: np.ndarray, index_name: str, col_prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(mat.copy())
    df.index.name = index_name
    df.columns = [f"{col_prefix}_{i}" for i in range(df.shape[1])]
    return df
