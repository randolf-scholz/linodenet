r"""Implementation of the Uniform distribution."""

__all__ = [
    # Classes
    "Uniform",
    "UniformParams",
]

from typing import Final, NamedTuple, override

import torch
from torch import Tensor

from .base import DistributionBase


class UniformParams(NamedTuple):
    r"""Uniform bounds and operations on the half-open interval `[low, high)`.

    Callers must provide floating-point bounds with the same shape, dtype, and
    device, satisfying `low < high` elementwise. Construction does not validate
    or broadcast bounds; use `Uniform` for that. The tuple is immutable, but its
    tensors remain mutable.
    """

    low: Tensor
    r"""Lower interval bound."""
    high: Tensor
    r"""Upper interval bound."""

    @property
    def mean(self) -> Tensor:
        r"""Mean of the distribution."""
        return (self.high + self.low) / 2

    @property
    def median(self) -> Tensor:
        r"""Median of the distribution."""
        return self.mean

    @property
    def mode(self) -> Tensor:
        r"""Raise `NotImplementedError` because the distribution has no unique mode."""
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        r"""Variance of the distribution."""
        return (self.high - self.low).square() / 12

    @property
    def stddev(self) -> Tensor:
        r"""Standard deviation of the distribution."""
        return self.variance.sqrt()

    def sample(self, num: int = 1, /) -> Tensor:
        r"""Draw samples with shape `(num, *low.shape)`."""
        shape = (num, *self.low.shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value: Tensor, /) -> Tensor:
        r"""Log density, with `-inf` outside `[low, high)`."""
        inside = (self.low <= value) & (value < self.high)
        log_density = -(self.high - self.low).log()
        return torch.where(inside, log_density, -torch.inf)

    def cdf(self, value: Tensor, /) -> Tensor:
        r"""Cumulative distribution function, clamped to `[0, 1]`."""
        result = (value - self.low) / (self.high - self.low)
        return result.clamp(min=0.0, max=1.0)

    def icdf(self, value: Tensor, /) -> Tensor:
        r"""Quantile function; callers must provide probabilities in `[0, 1]`."""
        return value * (self.high - self.low) + self.low

    def entropy(self) -> Tensor:
        r"""Differential entropy of the distribution."""
        return (self.high - self.low).log()


class Uniform(DistributionBase):
    r"""Uniform distribution on the half-open interval `[low, high)`.

    Bounds are broadcast to a common batch shape and must satisfy `low < high`
    elementwise. Samples are scalar events.
    """

    low: Tensor
    r"""Param: Lower interval bound."""
    high: Tensor
    r"""Param: Upper interval bound."""
    has_rsample: Final[bool] = True

    def __init__(self, low: Tensor | float, high: Tensor | float) -> None:
        low_tensor = torch.as_tensor(low)
        high_tensor = torch.as_tensor(high)
        low_broadcast, high_broadcast = torch.broadcast_tensors(low_tensor, high_tensor)

        if not torch.is_floating_point(low_broadcast):
            dtype = torch.get_default_dtype()
            low_broadcast = low_broadcast.to(dtype=dtype)
            high_broadcast = high_broadcast.to(dtype=dtype)

        if not torch.all(low_broadcast < high_broadcast):
            raise ValueError("Expected low < high elementwise.")

        super().__init__(batch_shape=low_broadcast.shape, event_shape=())
        self.register_buffer("low", low_broadcast)
        self.register_buffer("high", high_broadcast)

    @property
    def params(self) -> UniformParams:
        r"""Parameters referencing the current registered bounds."""
        return UniformParams(self.low, self.high)

    @property
    def mean(self) -> Tensor:
        return self.params.mean

    @property
    def median(self) -> Tensor:
        return self.params.median

    @property
    def mode(self) -> Tensor:
        return self.params.mode

    @property
    def variance(self) -> Tensor:
        return self.params.variance

    @property
    def stddev(self) -> Tensor:
        return self.params.stddev

    @override
    def sample(self, num: int = 1, /) -> Tensor:
        self.samples = self.params.sample(num)
        return self.samples

    @override
    def log_prob(self, value: Tensor, /) -> Tensor:
        self.log_probs = self.params.log_prob(value)
        return self.log_probs

    def cdf(self, value: Tensor, /) -> Tensor:
        return self.params.cdf(value)

    def icdf(self, value: Tensor, /) -> Tensor:
        return self.params.icdf(value)

    def entropy(self) -> Tensor:
        return self.params.entropy()
