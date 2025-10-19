import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset/toy")
TARGET_NAME = "target"
RANDOM_SEED = 123


def generate_toy_data(n_samples: int = 1000) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)

    cat_values = {
        "cat0": np.array(["A", "B", "C"]),
        "cat1": np.array(["X", "Y"]),
        "cat2": np.array(["K", "L", "M", "N"]),
        "cat3": np.array(["Yes", "No"]),
        "cat4": np.array(["Low", "Mid", "High"]),
    }

    num_features = ["num0", "num1", "num2", "num3", "num4"]

    cats = [
        rng.choice(values, size=n_samples)
        for values in cat_values.values()
    ]
    cats = np.stack(cats, axis=1)

    nums = rng.normal(0, 1, size=(n_samples, len(num_features))).astype(np.float32)

    # Simple synthetic binary target correlated with first categorical and numeric sum
    target_prob = (
        (cats[:, 0] == "A").astype(float)
        + 0.5 * (nums.sum(axis=1) > 0).astype(float)
    ) / 2.0
    y = (rng.random(n_samples) < target_prob).astype(np.float32)

    return {
        "categorical": cats,
        "numerical": nums,
        "target": y,
        "categorical_columns": list(cat_values.keys()),
        "numerical_columns": num_features,
        "cat_value_map": cat_values,
    }


def split_indices(n_samples: int, val_ratio: float = 0.1, test_ratio: float = 0.1):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    val_size = int(val_ratio / (val_ratio + test_ratio) * len(temp_idx))
    val_idx = temp_idx[:val_size]
    test_idx = temp_idx[val_size:]
    return train_idx, val_idx, test_idx


def save_array(path: Path, array: np.ndarray, allow_pickle: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=allow_pickle)


def save_dataset(data: Dict[str, np.ndarray]):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    cats = data["categorical"]
    nums = data["numerical"]
    y = data["target"]

    train_idx, val_idx, test_idx = split_indices(len(y))

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    for split_name, idx in splits.items():
        save_array(DATASET_DIR / f"X_cat_{split_name}.npy", cats[idx], allow_pickle=True)
        save_array(DATASET_DIR / f"X_num_{split_name}.npy", nums[idx].astype(np.float32))
        save_array(DATASET_DIR / f"y_{split_name}.npy", y[idx].astype(np.float32))

    cat_group_sizes: List[int] = []
    for col_idx in range(cats.shape[1]):
        unique_vals = np.unique(cats[train_idx, col_idx])
        cat_group_sizes.append(len(unique_vals))

    info = {
        "task_type": "binclass",
        "n_classes": 2,
        "n_num_features": int(nums.shape[1]),
        "n_cat_features": int(cats.shape[1]),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "cat_group_sizes": cat_group_sizes,
        "categorical_columns": data["categorical_columns"],
        "numeric_columns": data["numerical_columns"],
        "target": TARGET_NAME,
        "random_state": RANDOM_SEED,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }

    (DATASET_DIR / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    data = generate_toy_data()
    save_dataset(data)
    print(f"Toy dataset saved to {DATASET_DIR.resolve()}")


if __name__ == "__main__":
    main()
