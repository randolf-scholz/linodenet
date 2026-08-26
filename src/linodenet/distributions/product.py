r"""Implementation of the product distribution."""

__all__ = [
    "Product",
    "ProductSameFamily",
]

from typing import Final

import torch
from torch import Tensor

from linodenet.nn import ModuleSequence

from .base import DistributionBase


# TODO: Consider using TypeVarTuple if they ever become powerful enough.
class Product[D: DistributionBase](ModuleSequence[D]):
    r"""Represents the outer product of random distributions.

    .. math:: p(x₁，x₂，…，xₙ) = ∏ₖ pₖ(xₖ)

    Arguments:
        marginals: A list of distributions $p(Zᵢ)$.

    Example:
        >>> from linodenet.distributions import Uniform, Normal
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> d = Product(Uniform(0.0, 1.0), Normal(0.0, 1.0))
        >>> d.event_shape
        (torch.Size([]), torch.Size([]))
        >>> d.sample()
        (tensor([0.4963]), tensor([0.2072]))
    """

    has_rsample = True
    batch_shape: Final[tuple[int, ...]]
    event_shape: Final[tuple[tuple[int, ...], ...]]

    def __init__(self, *marginals: D) -> None:
        super().__init__(marginals)
        if len(self) < 1:
            raise ValueError("At least one marginal distribution is required.")

        self.batch_shape = torch.broadcast_shapes(*(m.batch_shape for m in self))
        self.event_shape = tuple(m.event_shape for m in self)

        if not all(m.batch_shape == self.batch_shape for m in self):
            raise ValueError(
                "All marginal distributions must have the same batch shape."
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
