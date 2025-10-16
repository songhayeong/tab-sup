import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ROOT = Path("dataset")
INPUT_CSV = DATASET_ROOT / "aimers_6th_final_train.csv"
OUTPUT_DIR = DATASET_ROOT / "aimers"
TARGET_COLUMN = "임신 성공 확률"
DROP_COLUMNS = ["ID"]
RANDOM_STATE = 777
VAL_RATIO = 0.1
TEST_RATIO = 0.1


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def compute_splits(n_samples: int) -> SplitIndices:
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_ratio,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def extract_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = [col for col in df.columns if col != TARGET_COLUMN]
    return categorical_cols, []


def to_numpy(df: pd.DataFrame, cols: List[str], dtype) -> np.ndarray:
    if not cols:
        return np.empty((len(df), 0), dtype=dtype)
    return df[cols].to_numpy().astype(dtype, copy=False)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    allow_pickle = array.dtype == np.object_
    np.save(path, array, allow_pickle=allow_pickle)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

    categorical_cols, numeric_cols = extract_feature_columns(df)

    df_cat = df[categorical_cols].astype(str) if categorical_cols else pd.DataFrame(index=df.index)
    df_num = df[numeric_cols] if numeric_cols else pd.DataFrame(index=df.index)

    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    splits = compute_splits(len(df))

    cat_group_sizes = [
        int(df_cat.iloc[splits.train][col].nunique())
        for col in categorical_cols
    ] if categorical_cols else []

    y_splits = {
        "train": y[splits.train],
        "val": y[splits.val],
        "test": y[splits.test],
    }

    X_cat_splits = {
        "train": to_numpy(df_cat.iloc[splits.train], categorical_cols, dtype=object),
        "val": to_numpy(df_cat.iloc[splits.val], categorical_cols, dtype=object),
        "test": to_numpy(df_cat.iloc[splits.test], categorical_cols, dtype=object),
    } if categorical_cols else None

    X_num_splits = (
        {
            "train": to_numpy(df_num.iloc[splits.train], numeric_cols, dtype=np.float32),
            "val": to_numpy(df_num.iloc[splits.val], numeric_cols, dtype=np.float32),
            "test": to_numpy(df_num.iloc[splits.test], numeric_cols, dtype=np.float32),
        }
        if numeric_cols
        else None
    )

    for split_name, values in y_splits.items():
        save_array(OUTPUT_DIR / f"y_{split_name}.npy", values)

    if X_cat_splits is not None:
        for split_name, values in X_cat_splits.items():
            save_array(OUTPUT_DIR / f"X_cat_{split_name}.npy", values.astype(np.object_))

    if X_num_splits is not None:
        for split_name, values in X_num_splits.items():
            save_array(OUTPUT_DIR / f"X_num_{split_name}.npy", values)

    info = {
        "task_type": "regression",
        "n_classes": None,
        "n_num_features": len(numeric_cols),
        "n_cat_features": len(categorical_cols),
        "train_size": int(len(splits.train)),
        "val_size": int(len(splits.val)),
        "test_size": int(len(splits.test)),
        "cat_group_sizes": cat_group_sizes,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    }
    info_path = OUTPUT_DIR / "info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Dataset prepared at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
