r"""Implementation of the Categorical distribution."""

__all__ = ["Categorical"]

from math import prod
from typing import Optional, override

import torch
from torch import Tensor

from linodenet.distributions.base import DistributionBase


class Categorical(DistributionBase):
    r"""A categorical distribution with fixed weights.

    Args:
        weights (..., D): The weights of the distribution. Requires $θ∈Δⁿ$.
        ndim: The number of dimensions of the distribution.
            If not given, `event_shape=weights.shape`, else `event_shape=weights.shape[-ndim:]`.
    """

    weights: Tensor
    r"""Param: The weights of the distribution."""

    def __init__(self, weights: Tensor, *, ndim: Optional[int] = None) -> None:
        event_shape = weights.shape[-ndim:] if ndim is not None else weights.shape
        batch_shape = weights.shape[: -len(self.event_shape)]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)
        assert weights.ndim >= 1, "Weights must be at least 1-dimensional."
        self.register_buffer("weights", weights)

    @classmethod
    def from_logits(cls, logits: Tensor, /) -> Categorical:
        r"""Create a categorical distribution from logits."""
        assert logits.ndim >= 1, "Logits must be at least 1-dimensional."
        raise NotImplementedError

    @classmethod
    def from_probs(cls, probs: Tensor, /) -> Categorical:
        r"""Create a categorical distribution from probabilities."""
        assert probs.ndim >= 1, "Probs must be at least 1-dimensional."
        assert torch.all(probs >= 0), "Probs must be non-negative."
        assert torch.all(probs <= 1), "Probs must be at most 1."
        assert torch.sum(probs, dim=-1) == 1, "Probs must sum to 1."
        raise NotImplementedError

    @property
    def num_components(self) -> int:
        return prod(self.event_shape)

    @property
    def logits(self) -> Tensor:
        return self.weights.log_softmax(dim=-1)

    @property
    def probs(self) -> Tensor:
        return self.weights.softmax(dim=-1)

    @override
    def sample(self, num: int = 1) -> Tensor:
        self.samples = torch.multinomial(self.probs, num, replacement=True)
        return self.samples

    @override
    def log_prob(self, samples: Tensor, /) -> Tensor:
        r"""Compute the negative log-likelihood of the samples."""
        self.log_probs = self.logits.gather(-1, samples).squeeze(-1)
        return self.log_probs

    def exclude(self, indices: Tensor, /) -> Categorical:
        r"""Condition the categorical distribution assuming the indices cannot occur.

        For a categorical distribution with parameters p∈Δⁿ, there is a nice fact:
        If the parameters are given by softmax transform: $p = σ(w)$,
        then $p₋ᵢ = σ(w₋ᵢ)$

        Proof:
            Let $q=p₋ᵢ$, w.l.o.g. $i=n$ is the last index. Then:

            .. math::
                qₖ &= pₖ / ∑_{j=1}^{n-1} pⱼ \\
                   &= (e^{wₖ}/∑_{i=1}^n e^{wᵢ}) / ∑_{j=1}^{n-1} (e^{wⱼ}/∑_{i=1}^n e^{wᵢ}) \\
                   &= e^{wₖ} / ∑_{j=1}^{n-1} e^{wⱼ}
                   &= σ(w₋ₙ)ₖ
        """
        weights = self.weights.clone()
        weights[indices] = float("-inf")
        return Categorical(weights)
