"""
Utility helpers that connect dataset preprocessing with the discrete graph
construction used by the diffusion model.
"""

from typing import Tuple

import torch

from .data import Dataset, Transformations, build_dataset, infer_block_group_sizes
from .graph_lib import Graph, get_graph


def prepare_dataset_and_graph(
        config,
        dataset_path,
        transformations: Transformations,
        device: torch.device,
        cache: bool = True,
) -> Tuple[Dataset, Graph]:
    """
    Loads and transforms the dataset, then instantiates the diffusion graph.

    For BlockUniform graphs, the categorical group sizes are inferred directly
    from the processed dataset so that the transition matrix aligns with the
    tabular feature layout.
    """
    dataset = build_dataset(dataset_path, transformations, cache)

    group_sizes = None
    if getattr(getattr(config, 'graph', None), 'type', None) == 'block_uniform':
        group_sizes = infer_block_group_sizes(dataset)

    graph = get_graph(config, device, group_sizes=group_sizes, dataset=dataset)
    return dataset, graph
