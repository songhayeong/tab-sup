import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = Path("data/shoppers")
RAW_CSV = RAW_DATA_DIR / "online_shoppers_intention.csv"
OUTPUT_DIR = Path("start/dataset/shoppers")
TARGET_COLUMN = "Revenue"

CAT_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

RANDOM_STATE = 777
VAL_RATIO = 0.1
TEST_RATIO = 0.1


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def compute_split_indices(
    y: np.ndarray, *, val_ratio: float, test_ratio: float, random_state: int
) -> SplitIndices:
    idx = np.arange(len(y))
    stratify = y.astype(int)
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=val_ratio + test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - relative_val_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=stratify[temp_idx],
    )
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def to_numpy(df: pd.DataFrame, columns: List[str], dtype) -> np.ndarray:
    if not columns:
        return np.empty((len(df), 0), dtype=dtype)
    return df[columns].to_numpy(dtype=dtype, copy=False)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    allow_pickle = array.dtype == object
    np.save(path, array, allow_pickle=allow_pickle)


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Could not locate raw dataset: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)

    # Encode target as integers {0, 1}
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    categorical_cols = [col for col in CAT_COLUMNS if col in df.columns]
    numeric_cols = [
        col
        for col in df.columns
        if col not in categorical_cols and col != TARGET_COLUMN
    ]

    df_categorical = (
        df[categorical_cols].astype(str) if categorical_cols else pd.DataFrame(index=df.index)
    )
    df_numeric = (
        df[numeric_cols].astype(np.float32) if numeric_cols else pd.DataFrame(index=df.index)
    )

    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    splits = compute_split_indices(
        y, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, random_state=RANDOM_STATE
    )

    y_splits = {
        "train": y[splits.train],
        "val": y[splits.val],
        "test": y[splits.test],
    }

    X_cat_splits = (
        {
            "train": to_numpy(df_categorical.iloc[splits.train], categorical_cols, object),
            "val": to_numpy(df_categorical.iloc[splits.val], categorical_cols, object),
            "test": to_numpy(df_categorical.iloc[splits.test], categorical_cols, object),
        }
        if categorical_cols
        else None
    )

    X_num_splits = (
        {
            "train": to_numpy(df_numeric.iloc[splits.train], numeric_cols, np.float32),
            "val": to_numpy(df_numeric.iloc[splits.val], numeric_cols, np.float32),
            "test": to_numpy(df_numeric.iloc[splits.test], numeric_cols, np.float32),
        }
        if numeric_cols
        else None
    )

    cat_group_sizes = (
        [
            int(df_categorical.iloc[splits.train][col].nunique())
            for col in categorical_cols
        ]
        if categorical_cols
        else []
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, values in y_splits.items():
        save_array(OUTPUT_DIR / f"y_{split}.npy", values.astype(np.float32))

    if X_cat_splits is not None:
        for split, values in X_cat_splits.items():
            save_array(OUTPUT_DIR / f"X_cat_{split}.npy", values.astype(object))

    if X_num_splits is not None:
        for split, values in X_num_splits.items():
            save_array(OUTPUT_DIR / f"X_num_{split}.npy", values.astype(np.float32))

    info = {
        "task_type": "binclass",
        "n_classes": 2,
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
    print(f"Prepared shoppers dataset at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
