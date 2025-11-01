import abc
from typing import Optional

import torch
import torch.nn.functional as F
import pandas as pd

from .catsample import sample_categorical
from .model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}'
            )
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm,"""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A Pytorch tensor representing the current state.
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A Pytorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x


@register_predictor(name="None")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        # return probs.argmax(dim=-1)
        return sample_categorical(probs)


def get_sampling_fn(config, graph, noise, batch_dims, eps, device, feature_fn=None, state_proj=lambda x: x):
    sampling_fn = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims,
        predictor=config.sampling.predictor,
        steps=config.sampling.steps,
        denoise=config.sampling.noise_removal,
        eps=eps,
        device=device,
        feature_fn=feature_fn,
        state_proj=state_proj,
    )

    return sampling_fn


def get_pc_sampler(
        graph,
        noise,
        batch_dims,
        predictor,
        steps,
        denoise=True,
        eps=1e-5,
        device=torch.device('cpu'),
        feature_fn=None,
        state_proj=lambda x: x,
):
    predictor = get_predictor(predictor)(graph, noise)
    projector = state_proj
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        base_score_fn = mutils.get_score_fn(model, train=False, sampling=True, discrete_dim=graph.dim)

        if feature_fn is not None:
            def sampling_score_fn(tokens, sigma):
                features = feature_fn(tokens, None)
                scores = base_score_fn(features, sigma)
                return scores[:, :graph.dim]
        else:
            def sampling_score_fn(tokens, sigma):
                scores = base_score_fn(tokens, sigma)
                return scores[:, :graph.dim]

        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)

        if denoise:
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)

        return x

    return pc_sampler


def make_block_uniform_feature_fn(graph, numeric_template: Optional[torch.Tensor] = None):
    if not hasattr(graph, "group_sizes"):
        return None
    group_sizes = list(graph.group_sizes)

    if numeric_template is not None and numeric_template.ndim == 1:
        numeric_template = numeric_template.unsqueeze(0)

    def feature_fn(tokens_global: torch.Tensor, numeric_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        local_tokens = graph.to_local(tokens_global).long()
        one_hots = [
            F.one_hot(local_tokens[..., idx], num_classes=size).float()
            for idx, size in enumerate(group_sizes)
        ]
        if one_hots:
            cat = torch.cat(one_hots, dim=-1)
        else:
            cat = torch.zeros(tokens_global.shape[0], 0, device=tokens_global.device)

        numeric_component: Optional[torch.Tensor]
        if numeric_values is not None:
            numeric_component = numeric_values
        elif numeric_template is not None:
            numeric_component = numeric_template.to(tokens_global.device)
            if numeric_component.shape[0] == 1:
                numeric_component = numeric_component.expand(tokens_global.shape[0], -1)
            elif numeric_component.shape[0] != tokens_global.shape[0]:
                raise ValueError("numeric_template batch dimension mismatch")
        else:
            numeric_component = None

        if numeric_component is None or numeric_component.numel() == 0:
            return cat
        if cat.numel() == 0:
            return numeric_component.float()
        return torch.cat([numeric_component.float(), cat], dim=-1)

    return feature_fn


def make_block_uniform_projector(graph):
    group_sizes = torch.tensor(graph.group_sizes, dtype=torch.long)

    def projector(tokens_global: torch.Tensor) -> torch.Tensor:
        local = graph.to_local(tokens_global).round().long()
        device = tokens_global.device
        sizes = group_sizes.to(device)
        local = torch.clamp(local, min=0)
        for idx, size in enumerate(sizes):
            local[..., idx] = local[..., idx].clamp(max=size.item() - 1)
        return graph.to_global(local)

    return projector


def decode_block_uniform_tokens(graph, tokens_global: torch.Tensor, encoder) -> torch.Tensor:
    local_tokens = graph.to_local(tokens_global).cpu().numpy()
    if encoder is None:
        return local_tokens
    if hasattr(encoder, 'named_steps') and 'ordinalencoder' in encoder.named_steps:
        ordinal = encoder.named_steps['ordinalencoder']
        categories = ordinal.categories_
    elif hasattr(encoder, 'categories_'):
        categories = encoder.categories_
    else:
        categories = None

    if categories is not None:
        for idx, cats in enumerate(categories):
            max_idx = len(cats) - 1
            local_tokens[:, idx] = local_tokens[:, idx].clip(0, max_idx)

    return encoder.inverse_transform(local_tokens)


def decoded_to_dataframe(decoded, columns=None, numeric=None, numeric_columns=None):
    frames = []
    if decoded is not None:
        frames.append(pd.DataFrame(decoded, columns=columns))
    if numeric is not None:
        if torch.is_tensor(numeric):
            numeric_np = numeric.detach().cpu().numpy()
        else:
            numeric_np = numeric
        if numeric_np.size > 0:
            if numeric_columns is None:
                numeric_columns = [f"num_{idx}" for idx in range(numeric_np.shape[1])]
            elif len(numeric_columns) != numeric_np.shape[1]:
                print(
                    f"Warning: numeric column metadata length {len(numeric_columns)} does not match "
                    f"generated data width {numeric_np.shape[1]}. Falling back to generic names."
                )
                numeric_columns = [f"num_{idx}" for idx in range(numeric_np.shape[1])]
            frames.append(pd.DataFrame(numeric_np, columns=numeric_columns))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def sample_block_uniform(
        model,
        graph,
        noise,
        num_samples,
        steps,
        predictor="analytic",
        denoise=True,
        eps=1e-5,
        device=None,
        feature_fn=None,
        state_proj=None,
        numeric_context: Optional[torch.Tensor] = None,
        numeric_clip: Optional[float] = None,
):
    if device is None:
        device = next(model.parameters()).device
    total_input_dim = getattr(model, "input_dim", None)
    if total_input_dim is None:
        if numeric_context is not None:
            total_input_dim = graph.dim + numeric_context.shape[-1]
        else:
            total_input_dim = graph.dim
    numeric_dim = max(total_input_dim - graph.dim, 0)

    if feature_fn is None:
        feature_fn = make_block_uniform_feature_fn(graph)
    if state_proj is None:
        state_proj = make_block_uniform_projector(graph)

    batch_dims = (num_samples,)

    if numeric_dim == 0:
        sampler = get_pc_sampler(
            graph=graph,
            noise=noise,
            batch_dims=batch_dims,
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=device,
            feature_fn=feature_fn,
            state_proj=state_proj,
        )
        tokens = sampler(model)
        return tokens

    if predictor != "analytic":
        raise NotImplementedError("Continuous sampling currently supports only the 'analytic' predictor.")

    numeric_state: torch.Tensor
    if numeric_context is not None:
        numeric_state = numeric_context.to(device)
        if numeric_state.ndim == 1:
            numeric_state = numeric_state.unsqueeze(0)
        if numeric_state.shape[0] == 1:
            numeric_state = numeric_state.expand(num_samples, -1).clone()
        elif numeric_state.shape[0] != num_samples:
            raise ValueError("numeric_context batch dimension mismatch")
        numeric_dim = numeric_state.shape[1]
    else:
        if hasattr(noise, "sigmas"):
            init_sigma = float(noise.sigmas.max().item()) if hasattr(noise.sigmas, "max") else float(noise.sigmas[1])
        else:
            init_sigma = 1.0
        numeric_state = torch.randn(num_samples, numeric_dim, device=device) * init_sigma

    if numeric_clip is not None and numeric_clip > 0:
        numeric_state = torch.clamp(numeric_state, -numeric_clip, numeric_clip)

    projector = state_proj
    base_score_fn = mutils.get_score_fn(model, train=False, sampling=True, discrete_dim=graph.dim)
    tokens = graph.sample_limit(*batch_dims).to(device)

    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    def match_numeric_shape(t: torch.Tensor) -> torch.Tensor:
        if t.ndim < numeric_state.ndim:
            t = t.view(*t.shape, *([1] * (numeric_state.ndim - t.ndim)))
        return t

    for idx in range(steps):
        t_curr = timesteps[idx]
        t_next = timesteps[idx + 1]
        t_batch = t_curr * torch.ones(num_samples, 1, device=device)
        sigma_cat_curr = noise(t_batch)[0]
        sigma_cat_next = noise(t_next * torch.ones(num_samples, 1, device=device))[0]
        if hasattr(noise, "numeric_schedule"):
            sigma_num_curr, _ = noise.numeric_schedule(t_batch)
            sigma_num_next, _ = noise.numeric_schedule(t_next * torch.ones(num_samples, 1, device=device))
        else:
            sigma_num_curr = sigma_cat_curr
            sigma_num_next = sigma_cat_next

        tokens = projector(tokens)
        features = feature_fn(tokens, numeric_state)
        scores = base_score_fn(features, sigma_cat_curr)
        discrete_scores = scores[:, :graph.dim]
        numeric_scores = scores[:, graph.dim:]

        dsigma_cat = sigma_cat_curr - sigma_cat_next
        tokens = projector(tokens)
        probs = graph.staggered_score(discrete_scores, dsigma_cat) * graph.transp_transition(tokens, dsigma_cat)
        tokens = sample_categorical(probs)

        sigma_sq_curr = sigma_num_curr ** 2
        sigma_sq_next = sigma_num_next ** 2
        delta_sigma_sq = torch.clamp(sigma_sq_curr - sigma_sq_next, min=0.0)

        delta_sigma_sq = match_numeric_shape(delta_sigma_sq)
        noise_scale = torch.sqrt(delta_sigma_sq + 1e-12)
        stochastic_term = noise_scale * torch.randn_like(numeric_state)
        numeric_state = numeric_state + delta_sigma_sq * numeric_scores + stochastic_term
        if numeric_clip is not None and numeric_clip > 0:
            numeric_state = torch.clamp(numeric_state, -numeric_clip, numeric_clip)

    if denoise:
        t_final = timesteps[-1] * torch.ones(num_samples, 1, device=device)
        sigma_cat_final = noise(t_final)[0]
        if hasattr(noise, "numeric_schedule"):
            sigma_num_final, _ = noise.numeric_schedule(t_final)
        else:
            sigma_num_final = sigma_cat_final
        tokens = projector(tokens)
        features = feature_fn(tokens, numeric_state)
        scores = base_score_fn(features, sigma_cat_final)
        discrete_scores = scores[:, :graph.dim]
        numeric_scores = scores[:, graph.dim:]
        tokens = projector(tokens)
        probs = graph.staggered_score(discrete_scores, sigma_cat_final) * graph.transp_transition(tokens, sigma_cat_final)
        tokens = sample_categorical(probs)
        sigma_sq_final = match_numeric_shape(sigma_num_final ** 2)
        numeric_state = numeric_state + sigma_sq_final * numeric_scores
        if numeric_clip is not None and numeric_clip > 0:
            numeric_state = torch.clamp(numeric_state, -numeric_clip, numeric_clip)

    return tokens, numeric_state
