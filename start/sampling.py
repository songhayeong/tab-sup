import abc
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
        base_score_fn = mutils.get_score_fn(model, train=False, sampling=True)

        if feature_fn is not None:
            def sampling_score_fn(tokens, sigma):
                features = feature_fn(tokens)
                return base_score_fn(features, sigma)
        else:
            sampling_score_fn = base_score_fn

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


def make_block_uniform_feature_fn(graph, numeric_template: torch.Tensor | None = None):
    if not hasattr(graph, "group_sizes"):
        return None
    group_sizes = list(graph.group_sizes)

    if numeric_template is not None:
        if numeric_template.ndim == 1:
            numeric_template = numeric_template.unsqueeze(0)

    def feature_fn(tokens_global: torch.Tensor) -> torch.Tensor:
        local_tokens = graph.to_local(tokens_global).long()
        one_hots = [
            F.one_hot(local_tokens[..., idx], num_classes=size).float()
            for idx, size in enumerate(group_sizes)
        ]
        cat = torch.cat(one_hots, dim=-1) if one_hots else torch.zeros(tokens_global.shape[0], 0, device=tokens_global.device)

        if numeric_template is not None:
            numeric = numeric_template.to(tokens_global.device)
            if numeric.shape[0] == 1:
                numeric = numeric.expand(tokens_global.shape[0], -1)
            elif numeric.shape[0] != tokens_global.shape[0]:
                raise ValueError("numeric_template batch dimension mismatch")
            return torch.cat([numeric, cat], dim=-1)
        return cat

    return feature_fn


def make_block_uniform_projector(graph):
    def projector(tokens_global: torch.Tensor) -> torch.Tensor:
        local = graph.to_local(tokens_global).round().long()
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


def decoded_to_dataframe(decoded, columns=None):
    df = pd.DataFrame(decoded, columns=columns)
    return df


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
        numeric_context: torch.Tensor | None = None,
):
    if device is None:
        device = next(model.parameters()).device
    if feature_fn is None:
        feature_fn = make_block_uniform_feature_fn(graph, numeric_template=numeric_context)
    if state_proj is None:
        state_proj = make_block_uniform_projector(graph)

    sampler = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(num_samples,),
        predictor=predictor,
        steps=steps,
        denoise=denoise,
        eps=eps,
        device=device,
        feature_fn=feature_fn,
        state_proj=state_proj,
    )
    return sampler(model)
