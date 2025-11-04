r"""Distributions base class."""

__all__ = [
    # ABCs & Protocols
    "DistributionBase",
    "DistributionList",
    "DistributionDict",
    "Marginalizable",
    # Classes
    "Product",
    "Mixture",
    "Flow",
    "ElementwiseFlow",
    "ConditionalFlow",
]

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Protocol, Self, SupportsIndex, overload

import torch
from torch import Tensor
from torch.distributions import Distribution

from linodenet.random.distributions.abstract import DistributionProto


class Marginalizable(DistributionProto, Protocol):
    r"""A protocol for marginalizable distributions."""

    @abstractmethod
    def marginalize(self, x: Tensor, /, *, dims: tuple[int, ...]) -> DistributionProto:
        r"""Marginalize over the given dimensions."""
        ...


class DistributionBase(Distribution):
    r"""Base class for distributions."""


class DistributionList(Distribution, Sequence[Distribution]):
    r"""A list of distributions, similar to `nn.ModuleList`."""

    bases: list[Distribution]

    def __init__(self, bases: Iterable[Distribution], /) -> None:
        super().__init__()
        self.bases = list(bases)
        if not self.bases:
            raise ValueError("At least one distribution must be provided.")

    def __len__(self) -> int:
        return len(self.bases)

    @overload
    def __getitem__(self, index: int, /) -> Distribution: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...
    def __getitem__(self, index: int | slice, /) -> Distribution | Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""Get the marginal distribution at the given index."""
        if isinstance(index, SupportsIndex):
            return self.bases[index.__index__()]
        return self.__class__(self.bases[index])


class DistributionDict(Distribution, Mapping[str, Distribution]):
    r"""A dictionary of distributions, similar to `nn.ModuleDict`."""

    bases: dict[str, Distribution]

    def __init__(self, bases: Mapping[str, Distribution], /) -> None:
        super().__init__()
        self.bases = dict(bases)
        if not self.bases:
            raise ValueError("At least one distribution must be provided.")

    def __len__(self) -> int:
        return len(self.bases)

    def __iter__(self) -> Iterator[str]:
        return iter(self.bases)

    def __getitem__(self, key: str, /) -> Distribution:
        r"""Get the marginal distribution at the given key."""
        return self.bases[key]


class Product(DistributionList):
    r"""Represents the outer product of random distributions.

    .. math:: p(x₁，x₂，…，xₙ) = ∏ₖ pₖ(xₖ)

    Arguments:
        marginals: A list of distributions $p(Zᵢ)$.

    Example:
        >>> d = Product(Uniform(0.0, 1.0), Normal(0.0, 1.0))
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([ 0.8969, -2.6717])
    """

    marginals: list[Distribution]

    has_rsample = True

    def __init__(self, marginals: Iterable[Distribution], /) -> None:
        super().__init__(marginals)
        self.marginals = self.bases
        batch_shape = torch.broadcast_shapes(*(m.batch_shape for m in marginals))
        self.marginals = [m.expand(batch_shape) for m in marginals]


class Mixture(DistributionList):
    r"""Creates a mixture of distributions for a random variable $X$.

    .. math:: p(x) = ∑ᵢ wᵢ⋅pᵢ(x)  \qq{with} wᵢ≥0 and ∑ᵢ wᵢ = 1

    References:
        https://wikipedia.org/wiki/Mixture_model
    """

    components: list[Distribution]
    weights: Tensor
    has_rsample = False

    def __init__(
        self,
        components: Iterable[Distribution],
        /,
        *,
        weights: Tensor,
    ) -> None:
        super().__init__(components)
        self.components = self.bases

        w = torch.as_tensor(weights)
        # normalize the weights
        if w.shape != (len(self),):
            raise ValueError(
                "The number of weights must match the number of components."
            )
        if not torch.all(w >= 0):
            raise ValueError("The weights must be non-negative.")
        self.weights = w / w.sum()

    def marginalize(self) -> "Mixture":
        r"""Return the marginal distribution.

        For a mixture, we have:

        .. math:: p(x) = ∑ₖ wₖ pₖ(x)
            ⟹ ∫ p(x) dxᵢ = ∫∑ₖwₖpₖ(x)dxᵢ = ∑ₖwₖ∫pₖ(x)dxᵢ= ∑ₖwₖp̃ₖ(x)
        """
        # return Mixture((d.marginalize() for d in self.distributions), self.weights)
        raise NotImplementedError

    @overload
    def __getitem__(self, index: int, /) -> Distribution: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...
    def __getitem__(self, index: int | slice, /) -> Distribution | Self:
        r"""Returns the sub-mixture at the given index."""
        if isinstance(index, SupportsIndex):
            return self.components[index.__index__()]
        return self.__class__(self.components[index], weights=self.weights[index])


class Flow(Distribution):
    r"""A distribution that is parameterized by a flow transformation."""


class ElementwiseFlow(Distribution):
    r"""A distribution that is parameterized by an element-wise flow transformation."""


class ConditionalFlow(Distribution):
    r"""A distribution that is parameterized by a conditional flow transformation."""
