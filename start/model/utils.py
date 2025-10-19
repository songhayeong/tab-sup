import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional


def get_model_fn(model, train=False):
    """
    Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: 'True' for training and 'False' for evaluation.
        mlm: If the input model is a mlm and models the base probability

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """
        Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
                for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()

            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma)

    return model_fn


def get_score_fn(model, train=False, sampling=False, discrete_dim: Optional[int] = None):
    if sampling:
        assert not train, "Must sample in eval mode"
        if discrete_dim is None:
            raise ValueError("discrete_dim must be provided when sampling")
    model_fn = get_model_fn(model, train=train)

    autocast_ctx = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext

    with autocast_ctx(dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext():
        def score_fn(x, sigma):
            sigma = sigma.reshape(-1)
            score = model_fn(x, sigma)

            if sampling:
                # when sampling return true score (not log used for training)
                discrete = score[:, :discrete_dim].exp() if discrete_dim > 0 else score[:, :0]
                numeric = score[:, discrete_dim:]
                return torch.cat([discrete, numeric], dim=-1)

            return score

    return score_fn
