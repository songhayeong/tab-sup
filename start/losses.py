"""

we use this loss function. because in graph_lib.py we implemented score-entropy. so it is quite intuitive idea.

this source came from sedd original repository.

"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from . import graph_lib
from .model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):
    group_sizes = list(getattr(graph, "group_sizes", []))

    def tokens_to_features(local_tokens: torch.Tensor) -> torch.Tensor:
        if local_tokens is None or local_tokens.numel() == 0 or not group_sizes:
            return torch.empty((local_tokens.shape[0], 0), device=local_tokens.device)
        one_hots = [
            F.one_hot(local_tokens[:, idx], num_classes=size).float()
            for idx, size in enumerate(group_sizes)
        ]
        return torch.cat(one_hots, dim=-1)

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        tokens = batch['tokens']
        numeric = batch['numeric']
        device = tokens.device
        batch_size = tokens.shape[0]

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
        sigma, dsigma = noise(t)

        tokens_global = graph.to_global(tokens)
        if perturbed_batch is None:
            perturbed_global = graph.sample_transition(tokens_global, sigma)
        else:
            perturbed_global = perturbed_batch

        perturbed_tokens_local = graph.to_local(perturbed_global)
        cat_features = tokens_to_features(perturbed_tokens_local)
        if numeric is not None and numeric.numel() > 0:
            features = torch.cat([numeric.float(), cat_features], dim=-1) if cat_features.numel() > 0 else numeric.float()
        else:
            features = cat_features

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(features, sigma)
        loss = graph.score_entropy(log_score, sigma, perturbed_global, tokens_global)
        loss = dsigma * loss

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer,
                    scaler,
                    params,
                    step,
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch).mean() / accum

            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()

                loss = total_loss
                total_loss = 0

        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn
