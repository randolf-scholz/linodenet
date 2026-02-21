r"""Implementation of Mixture Distribution."""

__all__ = ["Mixture", "MixtureSameFamily"]

from collections.abc import Iterable
from typing import Final, Self, SupportsIndex, overload

import torch
from torch import Tensor, nn

from linodenet.distributions.base import DistributionBase
from linodenet.distributions.categorical import Categorical
from linodenet.nn import ModuleSequence


class Mixture[D: DistributionBase](nn.Module):
    r"""Creates a mixture of distributions for a random variable $X$.

    .. math:: p(x) = ∑ᵢ wᵢ⋅pᵢ(x)  \qq{with} wᵢ≥0 and ∑ᵢ wᵢ = 1

    References:
        - https://wikipedia.org/wiki/Mixture_model
    """

    probs: Tensor
    has_rsample = False

    def __init__(
        self,
        components: Iterable[D],
        /,
        *,
        weights: Tensor,
    ) -> None:
        super().__init__()
        self.components = ModuleSequence(components)
        w = torch.as_tensor(weights)
        # normalize the weights
        if w.shape != (len(self),):
            raise ValueError(
                "The number of weights must match the number of components."
            )
        if not torch.all(w >= 0):
            raise ValueError("The weights must be non-negative.")
        self.probs = w / w.sum()

    def marginalize(self) -> Mixture:
        r"""Return the marginal distribution.

        For a mixture, we have:

        .. math:: p(x) = ∑ₖ wₖ pₖ(x)
            ⟹ ∫ p(x) dxᵢ = ∫∑ₖwₖpₖ(x)dxᵢ = ∑ₖwₖ∫pₖ(x)dxᵢ= ∑ₖwₖp̃ₖ(x)
        """
        # return Mixture((d.marginalize() for d in self.distributions), self.weights)
        raise NotImplementedError

    @overload
    def __getitem__(self, index: int, /) -> D: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...
    def __getitem__(self, index: int | slice, /) -> D | Self:
        r"""Returns the sub-mixture at the given index."""
        if isinstance(index, SupportsIndex):
            return self.components[index]
        return self.__class__(self.components[index], weights=self.probs[index])

    def __len__(self) -> int:
        r"""Returns the number of components in the mixture."""
        return len(self.components)


class MixtureSameFamily[D: DistributionBase](DistributionBase):
    r"""More efficient implementation of Mixture distribution.

    When all components are from the same class and this class supports multi-head
    operations, we can implement a more efficient mixture distribution.
    """

    mixture_distribution: Categorical
    r"""Distribution: The mixing distribution."""
    component_distribution: D
    r"""Distribution: The (multi-head) component distribution."""

    indices: Tensor
    r"""Buffer: the most recent sample indices."""
    latents: Tensor
    r"""Buffer: the most recent latent samples."""
    num_components: Final[int]
    r"""Param: The number of components."""

    def __init__(
        self,
        *,
        mixture_distribution: Categorical,
        component_distribution: D,
    ) -> None:
        super().__init__(
            batch_shape=component_distribution.batch_shape,
            event_shape=component_distribution.event_shape,
        )
        self.mixture_distribution = mixture_distribution
        self.component_distribution = component_distribution
        self.num_components = mixture_distribution.num_components
        self.register_buffer("indices", self.mixture_distribution.samples)
        self.register_buffer("latents", self.component_distribution.samples)

    def log_prob(self, x: Tensor, /) -> Tensor:
        r"""Compute the log-likelihood of the samples.

        .. math::  p(x) = ∑ₖ wₖ pₖ(x) ⟹ log p(x) = log ∑ₖ wₖ pₖ(x) = LSE(log wₖ + log pₖ(x))

        Args:
            x (..., *D): The samples to evaluate.

        Returns:
            log_prob (...): The log-likelihood of the samples.
        """
        logits = self.mixture_distribution.logits  # (H)

        base_ll = self.component_distribution.log_prob(x)  # (..., H)
        log_probs = (logits + base_ll).logsumexp(dim=-1)  # (H), (..., H) -> (...)
        # store buffers
        self.log_probs = log_probs
        return log_probs

    def sample(self, num: int = 1, /) -> Tensor:
        indices = self.mixture_distribution.sample(num)  # (N, H)
        latents = self.component_distribution.sample(num)  # (N, H, *D)
        samples = latents.gather(1, indices)  # (N, H, *D) -> (N, *D)
        # store buffers
        self.indices = indices
        self.latents = latents
        self.samples = samples
        return samples

    def sample_and_log_prob(self, num: int = 1, /) -> tuple[Tensor, Tensor]:
        indices, index_ll = self.mixture_distribution.sample_and_log_prob(num)
        latents, sample_ll = self.component_distribution.sample_and_log_prob(num)
        self.samples = latents.gather(0, indices)  # (H, ...) -> (...)
        # logits = self.mixture_distribution.logits

        self.log_probs = (index_ll + sample_ll).logsumexp(
            dim=-1
        )  # (H), (..., H) -> (...)
        return self.samples, self.log_probs
