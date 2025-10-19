"""
start package exposes dataset preprocessing utilities, graph constructors,
and pipeline helpers to wire SEDD-style diffusion to tabular data.
"""

from .data import (
    Dataset,
    Transformations,
    build_dataset,
    infer_block_group_sizes,
    prepare_dataloader,
    prepare_torch_dataloader,
)
from .graph_lib import get_graph, Graph, Uniform, Absorbing
from .pipeline import prepare_dataset_and_graph
from .train import train, Config as TrainConfig
from .sampling import (
    get_pc_sampler,
    make_block_uniform_feature_fn,
    make_block_uniform_projector,
    sample_block_uniform,
    decode_block_uniform_tokens,
    decoded_to_dataframe,
)

__all__ = [
    "Dataset",
    "Transformations",
    "build_dataset",
    "infer_block_group_sizes",
    "prepare_dataloader",
    "prepare_torch_dataloader",
    "get_graph",
    "Graph",
    "Uniform",
    "Absorbing",
    "prepare_dataset_and_graph",
    "train",
    "TrainConfig",
    "get_pc_sampler",
    "make_block_uniform_feature_fn",
    "make_block_uniform_projector",
    "sample_block_uniform",
    "decode_block_uniform_tokens",
    "decoded_to_dataframe",
]
