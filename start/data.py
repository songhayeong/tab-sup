"""
이 부분에서 실질적으로 tabular를 Load해야함 sedd에서는 llm의 scheme으로 전개되었지만,
tab-ddpm등을 활용해 어떤식으로 로드해서 graph_lib와 연결할 수 있는지 전략을 잘 잡아야함.

여기서 해야할 역할, 데이터셋 로드
토크나이징 전처리 -> 이거는 생각좀해보자
데이터를 모델 입력 형태로 정제 (block chunking)
pytorch dataloader 구성
분산 학습을 위한 샘플러 (optional)
학숩 루프에서 무한 반복 가능한 cycle loader 구현

여기서 꼭 해야할것
-> 원핫으로 전체를 다 transition matrix화 시켜야한다.

일단 tab-ddpm의 코드를 가져오되, 부족한 부분을 고려해보자
"""

import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from importlib.resources import path
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
import torch
import os
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from scipy.spatial.distance import cdist
from . import utils

from .utils import TaskType, load_json

ArrayDict = Dict[str, np.array]
TensorDict = Dict[str, torch.Tensor]

CAT_MISSING_VALUE = '__nan__'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent', 'new_category']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']
DEQUANT_DIST = Literal['uniform', 'beta', 'round', 'none']


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


def _extract_group_sizes_from_transform(transform: Optional[Any]) -> Optional[List[int]]:
    if transform is None:
        return None

    if hasattr(transform, 'group_sizes'):
        return list(transform.group_sizes)

    encoder = None
    if hasattr(transform, 'named_steps'):
        encoder = list(transform.named_steps.values())[-1]
    elif hasattr(transform, 'steps'):
        encoder = transform.steps[-1][1]
    else:
        encoder = transform

    if hasattr(encoder, 'group_sizes'):
        return list(encoder.group_sizes)
    if hasattr(encoder, 'categories_'):
        return [len(cats) for cats in encoder.categories_]

    return None


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]
    cat_group_sizes: Optional[List[int]] = None
    cat_columns: Optional[List[str]] = None
    dequantizer: Optional["Dequantizer"] = None

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> 'Dataset':
        dir_ = Path(dir_)
        splits = [k for k in ['train', 'val', 'test'] if dir_.joinpath(f'y_{k}.npy').exists()]

        def load(item) -> ArrayDict:
            return {
                x: cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle=True))  # type: ignore[code]
                for x in splits
            }

        if Path(dir_ / 'info.json').exists():
            info = utils.load_json(dir_ / 'info.json')
        else:
            info = None
        cat_group_sizes = None
        cat_columns = None
        if info is not None:
            cat_group_sizes = info.get('cat_group_sizes')
            if cat_group_sizes is not None:
                cat_group_sizes = list(cat_group_sizes)
            cat_columns = info.get('categorical_columns')
        return Dataset(
            load('X_num') if dir_.joinpath('X_num_train.npy').exists() else None,
            load('X_cat') if dir_.joinpath('X_cat_train.npy').exists() else None,
            load('y'),
            {},
            TaskType(info['task_type']),
            info.get('n_classes'),
            cat_group_sizes,
            cat_columns,
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_num['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_size(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(self,
                          predictions: Dict[str, np.ndarray],
                          prediction_type: Optional[str]) -> Dict[str, Any]:
        """
        이후 실제 gen된 애들에 대해 성능 측정 하는 코드 구현
        """
        pass


def change_val(dataset: Dataset, val_size: float = 0.2):
    # should be done before transformations

    y = np.concatenate([dataset.y['train'], dataset.y['val']], axis=0)

    ixs = np.arange(y.shape[0])
    if dataset.is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)

    dataset.y['train'] = y[train_ixs]
    dataset.y['val'] = y[val_ixs]

    if dataset.X_num is not None:
        X_num = np.concatenate([dataset.X_num['train'], dataset.X_num['val']], axis=0)
        dataset.X_num['train'] = X_num[train_ixs]
        dataset.X_num['val'] = X_num[val_ixs]

    if dataset.X_cat is not None:
        X_cat = np.concatenate([dataset.X_cat['train'], dataset.X_cat['val']], axis=0)
        dataset.X_cat['train'] = X_cat[train_ixs]
        dataset.X_cat['val'] = X_cat[val_ixs]

    return dataset


def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    """
    여기서 num-nan에 대해 drop rows를 할지 아니면 mean으로 impute할지는 생각해봐야할듯함.
    """
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: -v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        utils.raise_unknown('policy', policy)
    return dataset


def num_normalize(
        X: ArrayDict,
        normalization: Optional[Normalization],
        seed: Optional[int],
) -> Tuple[ArrayDict, Optional[Any]]:
    if normalization is None:
        return {k: v.astype(np.float32, copy=False) for k, v in X.items()}, None

    if normalization == 'standard':
        transformer = StandardScaler()
    elif normalization == 'quantile':
        n_quantiles = min(1000, X['train'].shape[0])
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution='normal',
            random_state=seed,
        )
    elif normalization == 'minmax':
        transformer = MinMaxScaler()
    else:
        utils.raise_unknown('normalization', normalization)
    transformer.fit(X['train'])
    transformed = {k: transformer.transform(v) for k, v in X.items()}
    transformed = {k: v.astype(np.float32, copy=False) for k, v in transformed.items()}
    return transformed, transformer


class Dequantizer:
    def __init__(self, dequant_dist: DEQUANT_DIST, int_col_idx_wrt_num: list, int_dequant_factor: float):
        self.dequant_dist = dequant_dist
        self.int_col_idx_wrt_num = int_col_idx_wrt_num
        self.int_dequant_factor = int_dequant_factor

    def transform(self, X):
        dtype = X.dtype
        X_int = X[:, self.int_col_idx_wrt_num]
        if self.dequant_dist == 'uniform':
            noise = np.random.uniform(size=X_int.shape).astype(dtype, copy=False)
            X[:, self.int_col_idx_wrt_num] = X_int + noise * self.int_dequant_factor
        elif self.dequant_dist == 'beta':
            noise = np.random.beta(
                self.int_dequant_factor,
                self.int_dequant_factor,
                size=X_int.shape,
            ).astype(dtype, copy=False) - dtype.type(0.5)
            X[:, self.int_col_idx_wrt_num] = X_int + noise
        elif self.dequant_dist in ['round', 'none']:
            pass
        return X

    def inverse_transform(self, X):
        dtype = X.dtype
        X_int = X[:, self.int_col_idx_wrt_num]
        if self.dequant_dist == 'uniform':
            X[:, self.int_col_idx_wrt_num] = np.floor(X_int).astype(dtype, copy=False)
        elif self.dequant_dist == 'beta':
            X[:, self.int_col_idx_wrt_num] = np.rint(X_int).astype(dtype, copy=False)
        elif self.dequant_dist == 'round':
            X[:, self.int_col_idx_wrt_num] = np.rint(X_int).astype(dtype, copy=False)
        elif self.dequant_dist == 'none':
            pass
        return X


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    normalized = {}
    missing_present = False
    for split, values in X.items():
        arr = values.astype(object, copy=True)
        mask = pd.isna(arr) | (arr == CAT_MISSING_VALUE)
        if mask.any():
            missing_present = True
            arr[mask] = CAT_MISSING_VALUE
        normalized[split] = arr

    if not missing_present:
        return X if policy is None else normalized

    if policy is None:
        return normalized
    if policy == 'most_frequent':
        imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
        imputer.fit(normalized['train'])
        return {k: cast(np.ndarray, imputer.transform(v)) for k, v in normalized.items()}
    if policy == 'new_category':
        return normalized
    utils.raise_unknown('categorical NaN policy', policy)
    return normalized


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
        X: ArrayDict,
        encoding: Optional[CatEncoding],
        y_train: Optional[np.ndarray],
        seed: Optional[int],
        return_encoder: bool = False
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X['train'])

        encoder = make_pipeline(oe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v).astype(np.int64) for k, v in X.items()}
        max_values = X['train'].max(axis=0)
        for part in X.keys():
            if part == 'train': continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = [
                    max_values[column_idx] + 1
                ]
        group_sizes = []
        for column_idx in range(X['train'].shape[1]):
            column_max = max(int(X[part][:, column_idx].max()) for part in X)
            group_sizes.append(column_max + 1)
        if return_encoder:
            setattr(encoder, 'group_sizes', group_sizes)
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode
    elif encoding == 'one-hot':
        encoder_kwargs = {
            "handle_unknown": "ignore",
            "dtype": np.float32,  # type: ignore[code]
        }
        if 'sparse_output' in sklearn.preprocessing.OneHotEncoder.__init__.__code__.co_varnames:  # type: ignore[attr-defined]
            encoder_kwargs["sparse_output"] = False  # type: ignore[code]
        else:
            encoder_kwargs["sparse"] = False  # type: ignore[code]
        ohe = sklearn.preprocessing.OneHotEncoder(**encoder_kwargs)  # type: ignore[code]
        encoder = make_pipeline(ohe)

        # encoder.steps.append({'ohe' , ohe})
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        group_sizes = [len(cats) for cats in encoder.named_steps['onehotencoder'].categories_]
        if return_encoder:
            setattr(encoder, 'group_sizes', group_sizes)
            return X, True, encoder
    else:
        utils.raise_unknown('encoding', encoding)
    if return_encoder:
        return X, True, encoder
    return (X, True)


def build_target(
        y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        utils.raise_unknown('policy', policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = 'mean'
    cat_nan_policy: Optional[CatNanPolicy] = 'new_category'
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'
    int_dequant_dist: Optional[DEQUANT_DIST] = None
    int_dequant_factor: float = 1.0


def transform_dataset(
        dataset: Dataset,
        transformations: Transformations,
        cache_dir: Optional[Path],
        return_transforms: bool = False
) -> Dataset:
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
                cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = utils.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num
    group_sizes = list(dataset.cat_group_sizes) if dataset.cat_group_sizes is not None else None
    dequantizer: Optional[Dequantizer] = None

    if X_num is not None and X_num['train'].size > 0:
        dequant_dist = transformations.int_dequant_dist
        if dequant_dist is not None and dequant_dist != 'none':
            train_vals = X_num['train']
            tol = 1e-6
            int_col_mask = np.all(np.abs(train_vals - np.round(train_vals)) < tol, axis=0)
            int_col_idx = np.where(int_col_mask)[0].tolist()
            if int_col_idx:
                dequantizer = Dequantizer(dequant_dist, int_col_idx, transformations.int_dequant_factor)
                X_num = {k: dequantizer.transform(v.astype(np.float32, copy=True)) for k, v in X_num.items()}

    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat_raw = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat_raw = cat_drop_rare(X_cat_raw, transformations.cat_min_frequency)

        if transformations.cat_encoding == 'one-hot':
            X_cat, _, cat_transform = cat_encode(
                deepcopy(X_cat_raw),
                None,
                dataset.y['train'],
                transformations.seed,
                return_encoder=True,
            )
            extracted = _extract_group_sizes_from_transform(cat_transform)
            if extracted is not None:
                group_sizes = extracted
            X_cat_one_hot, is_num, _ = cat_encode(
                X_cat_raw,
                transformations.cat_encoding,
                dataset.y['train'],
                transformations.seed,
                return_encoder=True,
            )
            if not is_num:
                raise RuntimeError("One-hot encoding expected to produce numeric features.")
            X_num = (
                X_cat_one_hot
                if X_num is None
                else {k: np.hstack([X_num[k], X_cat_one_hot[k]]) for k in X_num}
            )
        else:
            X_cat, is_num, cat_transform = cat_encode(
                X_cat_raw,
                transformations.cat_encoding,
                dataset.y['train'],
                transformations.seed,
                return_encoder=True
            )
            extracted = _extract_group_sizes_from_transform(cat_transform)
            if extracted is not None:
                group_sizes = extracted
            if is_num:
                X_num = (
                    X_cat
                    if X_num is None
                    else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
                )
                X_cat = None
    if group_sizes is None and dataset.X_cat is not None:
        group_sizes = dataset.get_category_size('train')

    if X_num is not None:
        X_num, num_transform = num_normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
        )

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(
        dataset,
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        y_info=y_info,
        cat_group_sizes=group_sizes,
        cat_columns=dataset.cat_columns,
        dequantizer=dequantizer,
    )
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        utils.dump_pickle((transformations, dataset), cache_path)
    # if return_transforms:
    # return dataset, num_transform, cat_transform
    return dataset


def build_dataset(
        path: Union[str, Path],
        transformations: Transformations,
        cache: bool
) -> Dataset:
    path = Path(path)
    dataset = Dataset.from_dir(path)
    return transform_dataset(dataset, transformations, path if cache else None)


def prepare_tensors(
        dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y


################
## DataLoader ##
################

class TabDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, split: Literal['train', 'val', 'test']):
        super().__init__()

        self.X_num = torch.from_numpy(dataset.X_num[split]).float() if dataset.X_num is not None else None
        self.X_cat = torch.from_numpy(dataset.X_cat[split]).long() if dataset.X_cat is not None else None
        self.y = torch.from_numpy(dataset.y[split]).float()

        assert self.y is not None
        assert self.X_num is not None or self.X_cat is not None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        numeric = self.X_num[idx] if self.X_num is not None else torch.empty(0, dtype=torch.float32)
        tokens = self.X_cat[idx] if self.X_cat is not None else torch.empty(0, dtype=torch.long)
        sample = {
            'tokens': tokens,
            'numeric': numeric,
            'y': self.y[idx],
        }
        return sample


def prepare_dataloader(
        dataset: Dataset,
        split: str,
        batch_size: int,
        num_workers: int = 0,
):
    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
    )
    while True:
        yield from loader


def prepare_torch_dataloader(
        dataset: Dataset,
        split: str,
        shuffle: bool,
        batch_size: int,
        num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return loader


def dataset_from_csv(paths: Dict[str, str], cat_features, target, T):
    assert 'train' in paths
    y = {}
    X_num = {}
    X_cat = {} if len(cat_features) else None
    for split in paths.keys():
        df = pd.read_csv(paths[split])
        y[split] = df[target].to_numpy().astype(float)
        if X_cat is not None:
            X_cat[split] = df[cat_features].to_numpy().astype(str)
        X_num[split] = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)

    dataset = Dataset(X_num, X_cat, y, {}, None, len(np.unique(y['train'])))
    return transform_dataset(dataset, T, None)


#########

def load_dataset_info(dataset_dir_name: str) -> Dict[str, Any]:
    path = Path("data/" + dataset_dir_name)
    info = utils.load_json(path / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    info['n_features'] = info['n_num_features'] + info['n_cat_features']
    info['path'] = path
    return info


def infer_block_group_sizes(dataset: Dataset) -> List[int]:
    """
    Helper to expose categorical group sizes for BlockUniform graphs.
    Prefers metadata collected during preprocessing, otherwise falls back
    to computing unique category counts from the raw categorical matrix.
    """
    if dataset.cat_group_sizes is not None:
        return dataset.cat_group_sizes
    if dataset.X_cat is not None:
        return dataset.get_category_size('train')
    raise ValueError("No categorical information available to infer BlockUniform group sizes.")
