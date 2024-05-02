r"""Distributions base class."""

__all__ = [
    # ABCs & Protocols
    "BaseDistribution",
    "DistributionProto",
    "ConditionalDistributionProto",
    "JointDistributionProto",
    "Marginalizable",
    # Classes
    "Product",
    "Mixture",
    "Flow",
    "ElementwiseFlow",
    "ConditionalFlow",
]

from abc import abstractmethod

import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
)
from typing_extensions import Protocol, SupportsInt, overload

from linodenet.types import Range


class DistributionProto(Protocol):
    r"""A protocol for distributions."""


class ConditionalDistributionProto(Protocol):
    r"""A protocol for conditional distributions."""


class JointDistributionProto(Protocol):
    r"""A protocol for joint distributions."""


class Marginalizable(DistributionProto, Protocol):
    r"""A protocol for marginalizable distributions."""

    @abstractmethod
    def marginalize(self, x, /, *, dims: tuple[int, ...]) -> "DistributionProto":
        r"""Marginalize over the given dimensions."""
        ...


class BaseDistribution(Distribution):
    r"""Base class for distributions."""


class Product(Distribution):
    r"""Represents the outer product of random distributions.

    .. math:: p(x₁，x₂，…，xₙ) = ∏ₖ pₖ(xₖ)

    Arguments:
        marginals: A list of distributions :math:`p(Z_i)`.

    Example:
        >>> d = Product(Uniform(0.0, 1.0), Normal(0.0, 1.0))
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([ 0.8969, -2.6717])
    """

    marginals: list[Distribution]

    has_rsample = True

    def __init__(self, *marginals: Distribution):
        super().__init__()
        batch_shape = torch.broadcast_shapes(*(m.batch_shape for m in marginals))
        self.marginals = [m.expand(batch_shape) for m in marginals]

    def __len__(self) -> int:
        r"""Get the number of marginal distributions."""
        return len(self.marginals)

    @overload
    def __getitem__(self, index: int, /) -> Distribution: ...
    @overload
    def __getitem__(self, index: Range[int], /) -> "Product": ...
    def __getitem__(self, index, /):
        r"""Get the marginal distribution at the given index."""
        if isinstance(index, SupportsInt):
            return self.marginals[int(index)]
        return Product(*self.marginals[index])


class Mixture(Distribution):
    r"""Creates a mixture of distributions for a random variable :math:`X`.

    .. math:: p(x) = ∑ᵢ wᵢ⋅pᵢ(x)  \qq{with} wᵢ≥0 and ∑ᵢ wᵢ = 1

    References:
        https://wikipedia.org/wiki/Mixture_model
    """

    has_rsample = False

    distributions: list[Distribution]
    weights: Tensor

    def __init__(self, distributions, weights):
        super().__init__()
        self.distributions = distributions
        self.weights = weights

    @overload
    def __getitem__(self, index: int, /) -> Distribution: ...
    @overload
    def __getitem__(self, index: Range[int], /) -> "Mixture": ...
    def __getitem__(self, index, /):
        r"""Returns the sub-mixture at the given index."""
        if isinstance(index, SupportsInt):
            return self.distributions[int(index)]
        return Mixture(self.distributions[index], self.weights[index])

    def marginalize(self):
        r"""Return the marginal distribution.

        For a mixture, we have:

        .. math:: p(x) = ∑ₖ wₖ pₖ(x)
            ⟹ ∫ p(x) dxᵢ = ∫∑ₖwₖpₖ(x)dxᵢ = ∑ₖwₖ∫pₖ(x)dxᵢ= ∑ₖwₖp̃ₖ(x)
        """
        # return Mixture((d.marginalize() for d in self.distributions), self.weights)
        raise NotImplementedError


class Flow(Distribution):
    r"""A distribution that is parameterized by a flow transformation."""


class ElementwiseFlow(Distribution):
    r"""A distribution that is parameterized by an element-wise flow transformation."""


class ConditionalFlow(Distribution):
    r"""A distribution that is parameterized by a conditional flow transformation."""
