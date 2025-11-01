"""
여기서부터 시작 forward에 대한 정의가 필요하기 때문이라 판단한다.
기존 sedd에서는 discrete 전이 상태 확률 모델 구현 및 score-entropy를 계산하는 로직이 이 파일에 구현되어있다.
"""
import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Sequence, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .data import Dataset


def get_graph(config, device, group_sizes: Optional[Sequence[int]] = None, dataset: Optional['Dataset'] = None):
    if config.graph.type == "uniform":
        tokens = getattr(config.graph, 'tokens', None)
        if tokens is None:
            raise ValueError("Uniform graph requires `graph.tokens` to be specified in the config.")
        return Uniform(tokens)
    elif config.graph.type == "absorb":
        tokens = getattr(config.graph, 'tokens', None)
        if tokens is None:
            raise ValueError("Absorbing graph requires `graph.tokens` to be specified in the config.")
        return Absorbing(tokens)
    elif config.graph.type == "block_absorb":
        if group_sizes is None and dataset is not None:
            from .data import infer_block_group_sizes
            group_sizes = infer_block_group_sizes(dataset)
        sizes = getattr(config.graph, 'group_sizes', None)
        if sizes is None:
            if group_sizes is None:
                raise ValueError("BlockAbsorb graph requires group_sizes from config, argument, or dataset metadata.")
            sizes = group_sizes
        return BlockAbsorbing(sizes)
    elif config.graph.type == "block_uniform":
        if group_sizes is None and dataset is not None:
            from .data import infer_block_group_sizes
            group_sizes = infer_block_group_sizes(dataset)
        sizes = getattr(config.graph, 'group_sizes', None)
        if sizes is None:
            if group_sizes is None:
                raise ValueError("BlockUniform graph requires group_sizes from config, argument, or dataset metadata.")
            sizes = group_sizes
        return BlockUniform(sizes)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))


class Graph(abc.ABC):
    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass

    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-the column of the rate matrix Q, where i is [B_1, ... , B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}
        """
        pass

    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        # return sample_categorical(transition_vector, method="hard")
        pass

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        # return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)
        pass

    def to_global(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens

    def to_local(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x) , which is approximated with
        e^{-{dsigma} E} score
        """
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def sample_rate(self, i, rate):
        base = F.one_hot(i, num_classes=self.dim).float()
        probs = base + rate
        probs = torch.clamp(probs, min=0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs_sum = torch.where(probs_sum == 0, torch.ones_like(probs_sum), probs_sum)
        probs = probs / probs_sum
        sampled = torch.multinomial(probs.reshape(-1, self.dim), 1, replacement=True).squeeze(-1)
        return sampled.view_as(i)

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim
        )

        # positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class BlockUniform(Graph):
    """
    Uniform transitions restricted to disjoint groups of indices.

    Intended to model tabular categorical variables where each feature constitutes
    a group and diffusion only mixes values within the same feature.
    """

    def __init__(self, group_sizes: Sequence[int]):
        if not group_sizes:
            raise ValueError("group_sizes must contain at least one positive integer")
        if any(x <= 0 for x in group_sizes):
            raise ValueError("Group sizes must be positive")

        self._group_sizes = torch.as_tensor(group_sizes, dtype=torch.long)
        self._group_sizes_list = [int(x) for x in group_sizes]
        self._dim = int(self._group_sizes.sum().item())
        group_starts = torch.cumsum(self._group_sizes, dim=0) - self._group_sizes
        self._group_starts = group_starts
        self._index_group = torch.repeat_interleave(
            torch.arange(len(group_sizes), dtype=torch.long),
            self._group_sizes
        )
        self._group_indices = [
            torch.arange(start.item(), start.item() + size.item(), dtype=torch.long)
            for start, size in zip(self._group_starts, self._group_sizes)
        ]
        self._uniform_cache = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def absorb(self) -> bool:
        return False

    @property
    def group_sizes(self) -> Sequence[int]:
        return self._group_sizes_list

    @property
    def group_starts(self) -> torch.Tensor:
        return self._group_starts

    @property
    def num_groups(self) -> int:
        return len(self._group_sizes_list)

    def _broadcast_buffers(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        return (
            self._group_sizes.to(device),
            self._group_starts.to(device),
            self._index_group.to(device),
        )

    def rate(self, i: torch.Tensor) -> torch.Tensor:
        device = i.device
        flat_i = i.reshape(-1)
        group_sizes, group_starts, index_group = self._broadcast_buffers(device)

        sample_group = index_group[flat_i]
        sample_sizes = group_sizes[sample_group]
        sample_starts = group_starts[sample_group]

        arange = torch.arange(self.dim, device=device)
        mask = (arange >= sample_starts[:, None]) & (arange < (sample_starts + sample_sizes)[:, None])
        inv_sizes = torch.where(sample_sizes > 0, 1.0 / sample_sizes.float(), torch.zeros_like(sample_sizes, dtype=torch.float32))
        edge = mask.float() * inv_sizes[:, None]
        diag_values = -(sample_sizes.float() - 1.0) / sample_sizes.float()
        edge.scatter_(-1, flat_i[:, None], diag_values[:, None])
        return edge.view(*i.shape, self.dim)

    def transp_rate(self, i: torch.Tensor) -> torch.Tensor:
        return self.rate(i)

    def to_global(self, local_tokens: torch.Tensor) -> torch.Tensor:
        if local_tokens.numel() == 0:
            return local_tokens
        return local_tokens + self._group_starts.to(local_tokens.device)

    def to_local(self, global_tokens: torch.Tensor) -> torch.Tensor:
        if global_tokens.numel() == 0:
            return global_tokens
        return global_tokens - self._group_starts.to(global_tokens.device)

    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        device = i.device
        sigma = sigma.to(device)
        flat_i = i.reshape(-1)
        sigma_expanded = sigma
        while sigma_expanded.ndim < i.ndim:
            sigma_expanded = sigma_expanded.unsqueeze(-1)
        sigma_expanded = sigma_expanded.expand_as(i)
        flat_sigma = sigma_expanded.reshape(-1)

        group_sizes, group_starts, index_group = self._broadcast_buffers(device)
        sample_group = index_group[flat_i]
        sample_sizes = group_sizes[sample_group]
        sample_starts = group_starts[sample_group]

        arange = torch.arange(self.dim, device=device)
        mask = (arange >= sample_starts[:, None]) & (arange < (sample_starts + sample_sizes)[:, None])
        base = torch.where(
            sample_sizes > 0,
            (1 - torch.exp(-flat_sigma)) / sample_sizes.float(),
            torch.zeros_like(flat_sigma, dtype=torch.float32),
        )
        trans = mask.float() * base[:, None]
        trans.scatter_(-1, flat_i[:, None], torch.zeros_like(flat_sigma)[:, None])
        row_sum = trans.sum(dim=-1, keepdim=True)
        diag = 1 - row_sum.squeeze(-1)
        trans.scatter_(-1, flat_i[:, None], diag[:, None])
        return trans.view(*i.shape, self.dim)

    def transp_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.transition(i, sigma)

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        probs = self.transition(i, sigma)
        flat_probs = probs.reshape(-1, self.dim)
        sampled = torch.multinomial(flat_probs, 1, replacement=True).squeeze(-1)
        return sampled.view_as(i)

    def sample_rate(self, i: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        base = F.one_hot(i, num_classes=self.dim).float()
        probs = base + rate
        probs = torch.clamp(probs, min=0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs_sum = torch.where(probs_sum == 0, torch.ones_like(probs_sum), probs_sum)
        probs = probs / probs_sum
        sampled = torch.multinomial(probs.reshape(-1, self.dim), 1, replacement=True).squeeze(-1)
        return sampled.view_as(i)

    def reverse_rate(self, i, score):
        base = self.transp_rate(i)
        if score.dim() == 2:
            score_expanded = torch.zeros_like(base)
            start_idx = 0
            for size in self._group_sizes_list:
                end_idx = start_idx + size
                score_expanded[:, :, start_idx:end_idx] = score[:, start_idx:end_idx].unsqueeze(1)
                start_idx = end_idx
        else:
            score_expanded = score

        normalized_rate = base * score_expanded
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def staggered_score(self, score: torch.Tensor, dsigma: torch.Tensor) -> torch.Tensor:
        batch = score.shape[0]
        result = torch.zeros(batch, self.num_groups, self.dim, device=score.device, dtype=score.dtype)
        exp_term = torch.exp(-dsigma).reshape(score.shape[0], 1)

        start_idx = 0
        for group_idx, size in enumerate(self._group_sizes_list):
            end_idx = start_idx + size
            sub_score = score[:, start_idx:end_idx]
            if size == 0:
                continue
            sub_sum = sub_score.sum(dim=-1, keepdim=True)
            factor = (exp_term - 1) / (exp_term * size)
            result[:, group_idx, start_idx:end_idx] = factor * sub_sum + sub_score / exp_term
            start_idx = end_idx

        return result

    def sample_limit(self, *batch_dims):
        if not batch_dims:
            batch_dims = (1,)
        device = self._group_sizes.device
        local = torch.stack(
            [
                torch.randint(0, size, batch_dims, device=device)
                for size in self._group_sizes_list
            ],
            dim=-1,
        )
        return self.to_global(local)

    def score_entropy(self, score, sigma, x, x0):
        device = score.device
        sigma = sigma.reshape(-1)
        entropy = torch.zeros(score.shape[0], device=device, dtype=score.dtype)

        for idx, size in enumerate(self._group_sizes_list):
            start = self._group_starts[idx].item()
            end = start + size

            uniform_graph = self._uniform_cache.get(size)
            if uniform_graph is None:
                uniform_graph = Uniform(size)
                self._uniform_cache[size] = uniform_graph

            score_slice = score[:, start:end]
            x_local = x[:, idx] - start
            x0_local = x0[:, idx] - start

            entropy += uniform_graph.score_entropy(score_slice, sigma, x_local, x0_local)

        return entropy


class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1

    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass

    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert

    def sample_rate(self, i, rate):
        base = F.one_hot(i, num_classes=self.dim).float()
        probs = base + rate
        probs = torch.clamp(probs, min=0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs_sum = torch.where(probs_sum == 0, torch.ones_like(probs_sum), probs_sum)
        probs = probs / probs_sum
        sampled = torch.multinomial(probs.reshape(-1, self.dim), 1, replacement=True).squeeze(-1)
        return sampled.view_as(i)

    def staggered_score(self, score, dsigma):
        score = score.clone()  # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        # positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy


class BlockAbsorbing(Graph):
    """
    Per-feature absorbing transitions. Each categorical group has its own mask state
    (absorbing value) that diffusion can transition into independently.
    """

    def __init__(self, group_sizes: Sequence[int]):
        if not group_sizes:
            raise ValueError("group_sizes must contain at least one positive integer")
        if any(x <= 0 for x in group_sizes):
            raise ValueError("Group sizes must be positive")

        base_sizes = torch.as_tensor(group_sizes, dtype=torch.long)
        self._base_sizes = base_sizes
        self._group_sizes = base_sizes + 1  # add absorbing state
        self._base_sizes_list = [int(x) for x in base_sizes.tolist()]
        self._group_sizes_list = [int(x) for x in self._group_sizes.tolist()]
        self._dim = int(self._group_sizes.sum().item())
        self._group_starts = torch.cumsum(self._group_sizes, dim=0) - self._group_sizes
        self._index_group = torch.repeat_interleave(
            torch.arange(len(group_sizes), dtype=torch.long),
            self._group_sizes
        )
        self._absorbing_cache = {}

    def _get_absorbing(self, base_size: int) -> Absorbing:
        absorbing = self._absorbing_cache.get(base_size)
        if absorbing is None:
            absorbing = Absorbing(base_size)
            self._absorbing_cache[base_size] = absorbing
        return absorbing

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def absorb(self) -> bool:
        return True

    @property
    def group_sizes(self) -> Sequence[int]:
        return self._group_sizes_list

    @property
    def base_group_sizes(self) -> Sequence[int]:
        return self._base_sizes_list

    @property
    def group_starts(self) -> torch.Tensor:
        return self._group_starts

    @property
    def num_groups(self) -> int:
        return len(self._group_sizes_list)

    def _broadcast_buffers(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        return (
            self._group_sizes.to(device),
            self._group_starts.to(device),
            self._index_group.to(device),
        )

    def to_global(self, local_tokens: torch.Tensor) -> torch.Tensor:
        if local_tokens.numel() == 0:
            return local_tokens
        return local_tokens + self._group_starts.to(local_tokens.device)

    def to_local(self, global_tokens: torch.Tensor) -> torch.Tensor:
        if global_tokens.numel() == 0:
            return global_tokens
        return global_tokens - self._group_starts.to(global_tokens.device)

    def rate(self, i: torch.Tensor) -> torch.Tensor:
        device = i.device
        _, group_starts, _ = self._broadcast_buffers(device)
        result = torch.zeros(*i.shape, self.dim, device=device)
        for idx, base_size in enumerate(self._base_sizes_list):
            start = group_starts[idx].item()
            size = self._group_sizes_list[idx]
            absorber = self._get_absorbing(base_size)
            local = i[..., idx] - start
            rate_local = absorber.rate(local)
            index = [slice(None)] * result.ndim
            index[-2] = idx
            index[-1] = slice(start, start + size)
            result[tuple(index)] = rate_local
        return result

    def transp_rate(self, i: torch.Tensor) -> torch.Tensor:
        device = i.device
        _, group_starts, _ = self._broadcast_buffers(device)
        result = torch.zeros(*i.shape, self.dim, device=device)
        for idx, base_size in enumerate(self._base_sizes_list):
            start = group_starts[idx].item()
            size = self._group_sizes_list[idx]
            absorber = self._get_absorbing(base_size)
            local = i[..., idx] - start
            rate_local = absorber.transp_rate(local)
            index = [slice(None)] * result.ndim
            index[-2] = idx
            index[-1] = slice(start, start + size)
            result[tuple(index)] = rate_local
        return result

    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        device = i.device
        sigma = sigma.to(device)
        sigma_expanded = sigma
        while sigma_expanded.ndim < i.ndim:
            sigma_expanded = sigma_expanded.unsqueeze(-1)
        sigma_expanded = sigma_expanded.expand_as(i)

        _, group_starts, _ = self._broadcast_buffers(device)
        result = torch.zeros(*i.shape, self.dim, device=device)
        for idx, base_size in enumerate(self._base_sizes_list):
            start = group_starts[idx].item()
            size = self._group_sizes_list[idx]
            local = i[..., idx] - start
            sigma_local = sigma_expanded[..., idx]

            flat_local = local.reshape(-1)
            flat_sigma = sigma_local.reshape(-1)
            trans_flat = torch.zeros(flat_local.size(0), size, device=device)
            absorbing_idx = size - 1
            abs_mask = flat_local == absorbing_idx
            if abs_mask.any():
                trans_flat[abs_mask, absorbing_idx] = 1.0
            active_mask = ~abs_mask
            if active_mask.any():
                stay = torch.exp(-flat_sigma[active_mask])
                move = 1 - stay
                active_local = flat_local[active_mask]
                trans_flat[active_mask, absorbing_idx] = move
                trans_flat[active_mask, active_local] = stay
            trans_local = trans_flat.view(*local.shape, size)
            index = [slice(None)] * result.ndim
            index[-2] = idx
            index[-1] = slice(start, start + size)
            result[tuple(index)] = trans_local
        return result

    def transp_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        device = i.device
        sigma = sigma.to(device)
        sigma_expanded = sigma
        while sigma_expanded.ndim < i.ndim:
            sigma_expanded = sigma_expanded.unsqueeze(-1)
        sigma_expanded = sigma_expanded.expand_as(i)

        _, group_starts, _ = self._broadcast_buffers(device)
        result = torch.zeros(*i.shape, self.dim, device=device)
        for idx, base_size in enumerate(self._base_sizes_list):
            start = group_starts[idx].item()
            size = self._group_sizes_list[idx]
            absorber = self._get_absorbing(base_size)
            local = i[..., idx] - start
            sigma_local = sigma_expanded[..., idx]
            trans_local = absorber.transp_transition(local, sigma_local)
            index = [slice(None)] * result.ndim
            index[-2] = idx
            index[-1] = slice(start, start + size)
            result[tuple(index)] = trans_local
        return result

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        device = i.device
        sigma = sigma.to(device)
        sigma_expanded = sigma
        while sigma_expanded.ndim < i.ndim:
            sigma_expanded = sigma_expanded.unsqueeze(-1)
        sigma_expanded = sigma_expanded.expand_as(i)

        local = self.to_local(i)
        result_local = torch.empty_like(local)
        for idx, base_size in enumerate(self._base_sizes_list):
            absorber = self._get_absorbing(base_size)
            sigma_local = sigma_expanded[..., idx]
            result_local[..., idx] = absorber.sample_transition(local[..., idx], sigma_local)
        return self.to_global(result_local)

    def reverse_rate(self, i: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        base = self.transp_rate(i)
        if score.dim() == 2:
            score_expanded = torch.zeros_like(base)
            start_idx = 0
            for size in self._group_sizes_list:
                end_idx = start_idx + size
                score_expanded[:, :, start_idx:end_idx] = score[:, start_idx:end_idx].unsqueeze(1)
                start_idx = end_idx
        else:
            score_expanded = score

        normalized_rate = base * score_expanded
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def staggered_score(self, score: torch.Tensor, dsigma: torch.Tensor) -> torch.Tensor:
        batch = score.shape[0]
        result = torch.zeros(batch, self.num_groups, self.dim, device=score.device, dtype=score.dtype)
        dsigma_expanded = dsigma.reshape(score.shape[0], 1)

        start_idx = 0
        for group_idx, base_size in enumerate(self._base_sizes_list):
            size = self._group_sizes_list[group_idx]
            end_idx = start_idx + size
            sub_score = score[:, start_idx:end_idx]
            absorber = self._get_absorbing(base_size)
            sub_result = absorber.staggered_score(sub_score, dsigma_expanded.squeeze(-1))
            result[:, group_idx, start_idx:end_idx] = sub_result
            start_idx = end_idx

        return result

    def sample_limit(self, *batch_dims):
        if not batch_dims:
            batch_dims = (1,)
        device = self._group_sizes.device
        local = torch.stack(
            [
                absorber.sample_limit(*batch_dims).to(device)
                for absorber in (self._get_absorbing(size) for size in self._base_sizes_list)
            ],
            dim=-1,
        )
        return self.to_global(local)

    def score_entropy(self, score, sigma, x, x0):
        device = score.device
        sigma = sigma.reshape(-1)
        entropy = torch.zeros(score.shape[0], device=device, dtype=score.dtype)

        for idx, base_size in enumerate(self._base_sizes_list):
            start = self._group_starts[idx].item()
            size = self._group_sizes_list[idx]
            absorber = self._get_absorbing(base_size)

            score_slice = score[:, start:start + size]
            x_local = x[:, idx] - start
            x0_local = x0[:, idx] - start

            entropy += absorber.score_entropy(score_slice, sigma, x_local, x0_local)

        return entropy
