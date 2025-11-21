r"""Empirical and Dirac distributions."""

__all__ = [
    "Dirac",
    "Empirical",
]

from typing import Final, Optional

import torch
from torch import Tensor

from linodenet.distributions.base import DistributionBase


class Empirical(DistributionBase):
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

    def __init__(self, values: Tensor, /, ndim: Optional[int] = None) -> None:
        r"""Initialize the empirical distribution.

        Args:
            values: (N, *D) or (*Bs, N, *D): The dataset that defines the empirical distribution.
            ndim: The number of dimensions of each data point. If not given,
            it is assumed that unbatched data is given, i.e., `ndim=values.ndim - 1`.
        """
        assert values.ndim >= 1, "The data must have at least one dimension."
        n = values.ndim - 1 if ndim is None else ndim
        assert 0 <= n < values.ndim, "ndim must between 0 and values.ndim."
        batch_shape = values.shape[: -(n + 1)]
        num_samples = values.shape[-(n + 1)]
        event_shape = values.shape[-n : n and None]  # last n elements (n maybe 0)
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)
        self.data = values
        self.num_samples = num_samples
        self.ndims = len(self.event_shape)
        self.dims = tuple(range(-self.ndims, 0))

        if batch_shape != ():
            raise NotImplementedError(
                "Empirical distribution with batch shape is not implemented."
            )

    def sample(
        self,
        num: int = 1,
    ) -> Tensor:
        r"""Sample from the empirical distribution."""
        idx = torch.randint(self.num_samples, size=(num,), device=self.data.device)
        self.samples = self.data[idx]  # TODO: support batch shape
        return self.samples

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
        self.log_probs = torch.where(mask, torch.inf, -torch.inf)
        return self.log_probs


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

    def sample(self, num: int = 1) -> Tensor:
        r"""Sample from the Dirac distribution."""
        self.samples = self.data.expand(*(num,), *self.event_shape)
        return self.samples

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log probability of the Dirac distribution.

        Formally, we set δ(0) = ∞ and δ(x) = 0 for x ≠ 0.

        .. Signature: ``[...] -> [...]``.
        """
        # NOTE: list[-n: n and None] fancy way to get last n elements for n≥0
        assert value.shape[-self.ndims : self.ndims and None] == self.event_shape
        mask = value == self.data  # (..., *D), (*D) -> (..., *D)
        mask = mask.all(dim=self.dims)  # (..., *D) -> (...)
        self.log_probs = torch.where(mask, torch.inf, -torch.inf)
        return self.log_probs
