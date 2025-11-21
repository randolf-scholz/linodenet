r"""Implementation of the product distribution."""

__all__ = [
    "Product",
    "ProductSameFamily",
]

from collections.abc import Iterable
from typing import Final

import torch
from torch import Tensor

from linodenet.distributions.base import DistributionBase
from linodenet.torch_generics import ModuleSequence


class Product[D: DistributionBase](ModuleSequence[D]):
    r"""Represents the outer product of random distributions.

    .. math:: p(x₁，x₂，…，xₙ) = ∏ₖ pₖ(xₖ)

    Arguments:
        marginals: A list of distributions $p(Zᵢ)$.

    Example:
        >>> from torch.distributions import Uniform, Normal
        >>> d = Product(Uniform(0.0, 1.0), Normal(0.0, 1.0))
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([ 0.8969, -2.6717])
    """

    has_rsample = True
    batch_shape: Final[tuple[int, ...]]
    event_shape: Final[tuple[int, ...]]

    def __init__(self, marginals: Iterable[D], /) -> None:
        super().__init__(marginals)
        self.batch_shape = torch.broadcast_shapes(*(m.batch_shape for m in self))
        assert len(self) >= 1, "At least one marginal distribution is required."
        first_marginal = self[0]
        self.event_shape = first_marginal.event_shape
        assert all(m.event_shape == self.event_shape for m in self), (
            "All marginal distributions must have the same event shape."
        )

    def sample(self, num_samples: int = 1, /) -> tuple[Tensor, ...]:
        r"""Sample from the product distribution."""
        return tuple(marginal.sample(num_samples) for marginal in self)

    def log_prob(self, value: tuple[Tensor, ...], /) -> Tensor:
        r"""Compute the log probability of the given samples.

        .. math:: log p(x₁ , ..., xₙ) = ∑ₖ log pₖ(xₖ)
        """
        log_probs = [
            marginal.log_prob(v) for marginal, v in zip(self, value, strict=True)
        ]
        return torch.stack(log_probs, dim=0).sum(dim=0)


class ProductSameFamily:
    r"""More efficient product distribution for same-family distributions."""
