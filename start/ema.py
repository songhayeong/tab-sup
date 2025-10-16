from typing import Iterable, List, Optional

import torch
from torch import nn


class ExponentialMovingAverage:
    """
    Minimal EMA tracker compatible with the interface expected by losses.get_step_fn.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], decay: float):
        self.decay = decay
        self._shadow_params: List[torch.Tensor] = []
        self._params: List[nn.Parameter] = []
        self._backup_params: Optional[List[torch.Tensor]] = None

        for param in parameters:
            if not param.requires_grad:
                continue
            self._params.append(param)
            self._shadow_params.append(param.detach().clone())

    @torch.no_grad()
    def update(self, parameters: Iterable[nn.Parameter]) -> None:
        for param, shadow in zip(parameters, self._shadow_params, strict=False):
            if not param.requires_grad:
                continue
            shadow.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def store(self, parameters: Iterable[nn.Parameter]) -> None:
        self._backup_params = [param.detach().clone() for param in parameters]

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[nn.Parameter]) -> None:
        for param, shadow in zip(parameters, self._shadow_params, strict=False):
            if not param.requires_grad:
                continue
            param.data.copy_(shadow)

    @torch.no_grad()
    def restore(self, parameters: Iterable[nn.Parameter]) -> None:
        if self._backup_params is None:
            return
        for param, backup in zip(parameters, self._backup_params, strict=False):
            param.data.copy_(backup)
        self._backup_params = None

    def state_dict(self) -> dict:
        return {
            'decay': self.decay,
            'shadow_params': [param.clone() for param in self._shadow_params],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict['decay']
        shadow_params = state_dict['shadow_params']
        if len(shadow_params) != len(self._shadow_params):
            raise ValueError("Mismatch in shadow parameter lengths when loading EMA state")
        for current, loaded in zip(self._shadow_params, shadow_params, strict=False):
            current.copy_(loaded)
