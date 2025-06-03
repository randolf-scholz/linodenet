r"""Empirical and Dirac distributions."""

__all__ = [
    "Dirac",
    "Empirical",
]


from collections.abc import Sequence
from typing import Final

import torch
from torch import Tensor
from torch.distributions import Distribution


class Empirical(Distribution):
    r"""The empirical distribution.

    .. math:: p(x) = \frac{1}{n} ∑_{i=1}^n δ(x - xᵢ)
    """

    data: Tensor  # shape: (N, *D)
    r"""CONST: The dataset that defines the empirical distribution."""
    dims: Final[tuple[int, ...]]  # (-D, ..., -1)
    r"""CONST: The dimensions of the data."""
    num_samples: Final[int]  # N
    r"""CONST: The number of samples in the dataset."""
    ndims: Final[int]  # D
    r"""CONST: The number of dimensions of the data."""

    def __init__(self, values: Tensor, /) -> None:
        r"""Initialize the empirical distribution."""
        super().__init__(event_shape=values.shape[1:])
        assert values.ndim >= 1, "The data must have at least one dimension."
        self.data = values  # shape: (n, ...)
        self.num_samples = values.shape[0]
        self.ndims = len(self.event_shape)
        self.dims = tuple(range(-self.ndims, 0))

    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        r"""Sample from the empirical distribution."""
        idx = torch.randint(
            self.num_samples, size=sample_shape, device=self.data.device
        )
        return self.data[idx]

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log probability of the empirical distribution.

        Formally, we set δ(0) = ∞ and δ(x) = 0 for x ≠ 0.

        .. Signature: ``[..., *D] -> [...]``.
        """
        # (..., *D), (N, *D) -> (..., N, *D)
        # NOTE: list[-n: n and None] fancy way to get last n elements for n≥0
        assert value.shape[-self.ndims : self.ndims and None] == self.event_shape
        # unsqueeze to allow broadcasting
        value = value.unsqueeze(dim=-(self.ndims + 1))  # (..., *D) -> (..., 1, *D)
        mask = value == self.data  # (..., 1, *D), (N, *D) -> (..., N, *D)
        # perform all mask over last *D dimensions, then
        mask = mask.all(dim=self.dims).any(dim=-1)
        return torch.where(mask, torch.inf, -torch.inf)


class Dirac(Empirical):
    r"""The Dirac distribution.

    .. math:: p(x) = δ(x - x₀)
    """

    data: Tensor  # shape: (*D)
    r"""CONST: The value that defines the Dirac distribution."""

    def __init__(self, value: Tensor, /) -> None:
        r"""Initialize the Dirac distribution."""
        super().__init__(value.unsqueeze(dim=0))
        assert self.num_samples == 1, "Dirac distribution must have exactly one sample."
        # overwrite data with correct squeezed shape.
        self.data = self.data.squeeze(dim=0)

    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        r"""Sample from the Dirac distribution."""
        return self.data.expand(*sample_shape, *self.event_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log probability of the Dirac distribution.

        Formally, we set δ(0) = ∞ and δ(x) = 0 for x ≠ 0.

        .. Signature: ``[...] -> [...]``.
        """
        # NOTE: list[-n: n and None] fancy way to get last n elements for n≥0
        assert value.shape[-self.ndims : self.ndims and None] == self.event_shape
        mask = value == self.data  # (..., *D), (*D) -> (..., *D)
        mask = mask.all(dim=self.dims)  # (..., *D) -> (...)
        return torch.where(mask, torch.inf, -torch.inf)
