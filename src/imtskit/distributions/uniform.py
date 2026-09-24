r"""Implementation of the Uniform distribution."""

__all__ = [
    # Classes
    "Uniform",
    # functions
    "uniform_mean",
    "uniform_median",
    "uniform_mode",
    "uniform_variance",
    "uniform_stddev",
    "uniform_cdf",
    "uniform_icdf",
    "uniform_entropy",
    "uniform_log_prob",
    "uniform_sample",
]

from typing import Final, override

import torch
from torch import Tensor

from .base import DistributionBase

type UniformParams = tuple[Tensor, Tensor]
r"""Parameters of the Uniform distribution (low, high)."""


def uniform_mean(params: UniformParams, /) -> Tensor:
    low, high = params
    return (high + low) / 2


def uniform_median(params: UniformParams, /) -> Tensor:
    return uniform_mean(params)


def uniform_mode(params: UniformParams, /) -> Tensor:
    raise NotImplementedError


def uniform_variance(params: UniformParams, /) -> Tensor:
    low, high = params
    return (high - low).pow(2) / 12


def uniform_stddev(params: UniformParams, /) -> Tensor:
    return uniform_variance(params).sqrt()


def uniform_sample(params: UniformParams, num: int, /) -> Tensor:
    low, high = params
    shape = (num, *low.shape)
    rand = torch.rand(shape, dtype=low.dtype, device=low.device)
    return low + rand * (high - low)


def uniform_log_prob(params: UniformParams, value: Tensor, /) -> Tensor:
    low, high = params
    inside = (low <= value) & (value < high)
    log_density = -(high - low).log()
    return torch.where(inside, log_density, -torch.inf)


def uniform_cdf(params: UniformParams, value: Tensor, /) -> Tensor:
    low, high = params
    result = (value - low) / (high - low)
    return result.clamp(min=0.0, max=1.0)


def uniform_icdf(params: UniformParams, value: Tensor, /) -> Tensor:
    low, high = params
    return value * (high - low) + low


def uniform_entropy(params: UniformParams, /) -> Tensor:
    low, high = params
    return (high - low).log()


class Uniform(DistributionBase):
    r"""Uniform distribution on the half-open interval $[low, high)$."""

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
    def params(self) -> tuple[Tensor, Tensor]:
        return self.low, self.high

    @property
    def mean(self) -> Tensor:
        return uniform_mean(self.params)

    @property
    def median(self) -> Tensor:
        return uniform_median(self.params)

    @property
    def mode(self) -> Tensor:
        return uniform_mode(self.params)

    @property
    def variance(self) -> Tensor:
        return uniform_variance(self.params)

    @property
    def stddev(self) -> Tensor:
        return uniform_stddev(self.params)

    @override
    def sample(self, num: int = 1, /) -> Tensor:
        self.samples = uniform_sample(self.params, num)
        return self.samples

    @override
    def log_prob(self, value: Tensor, /) -> Tensor:
        self.log_probs = uniform_log_prob(self.params, value)
        return self.log_probs

    def cdf(self, value: Tensor, /) -> Tensor:
        return uniform_cdf(self.params, value)

    def icdf(self, value: Tensor, /) -> Tensor:
        return uniform_icdf(self.params, value)

    def entropy(self) -> Tensor:
        return uniform_entropy(self.params)
