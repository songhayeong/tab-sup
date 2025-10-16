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
]
