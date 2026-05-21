r"""Implementation of the univariate Normal distribution."""

__all__ = [
    # Classes
    "Normal",
    # functions
    "normal_mean",
    "normal_median",
    "normal_mode",
    "normal_variance",
    "normal_stddev",
    "normal_cdf",
    "normal_icdf",
    "normal_entropy",
    "normal_log_prob",
    "normal_sample",
]

import math
from typing import Final, override

import torch
from torch import Tensor

from .base import DistributionBase

type NormalParams = tuple[Tensor, Tensor]
r"""Parameters of the Normal distribution $(loc, scale)$."""

_SQRT_2: Final[float] = math.sqrt(2.0)
_HALF_LOG_2PI: Final[float] = 0.5 * math.log(2.0 * math.pi)
_HALF_LOG_2PIE: Final[float] = 0.5 * math.log(2.0 * math.pi * math.e)


def normal_mean(params: NormalParams, /) -> Tensor:
    loc, _ = params
    return loc


def normal_median(params: NormalParams, /) -> Tensor:
    return normal_mean(params)


def normal_mode(params: NormalParams, /) -> Tensor:
    return normal_mean(params)


def normal_variance(params: NormalParams, /) -> Tensor:
    _, scale = params
    return scale.square()


def normal_stddev(params: NormalParams, /) -> Tensor:
    _, scale = params
    return scale


def normal_sample(params: NormalParams, num: int, /) -> Tensor:
    loc, scale = params
    shape = (num, *loc.shape)
    eps = torch.randn(shape, dtype=loc.dtype, device=loc.device)
    return loc + eps * scale


def normal_log_prob(params: NormalParams, value: Tensor, /) -> Tensor:
    loc, scale = params
    z = (value - loc) / scale
    return -0.5 * z.square() - scale.log() - _HALF_LOG_2PI


def normal_cdf(params: NormalParams, value: Tensor, /) -> Tensor:
    loc, scale = params
    z = (value - loc) / (scale * _SQRT_2)
    return 0.5 * (1.0 + torch.erf(z))


def normal_icdf(params: NormalParams, value: Tensor, /) -> Tensor:
    loc, scale = params
    return loc + scale * torch.erfinv(2.0 * value - 1.0) * _SQRT_2


def normal_entropy(params: NormalParams, /) -> Tensor:
    _, scale = params
    return scale.log() + _HALF_LOG_2PIE


class Normal(DistributionBase):
    r"""Univariate normal distribution parameterized by $loc$ and $scale$."""

    loc: Tensor
    r"""Param: Mean of the distribution."""
    scale: Tensor
    r"""Param: Standard deviation of the distribution."""
    has_rsample: Final[bool] = True

    def __init__(self, loc: Tensor | float, scale: Tensor | float) -> None:
        loc_tensor = torch.as_tensor(loc)
        scale_tensor = torch.as_tensor(scale)
        loc_broadcast, scale_broadcast = torch.broadcast_tensors(
            loc_tensor, scale_tensor
        )

        if not torch.is_floating_point(loc_broadcast):
            dtype = torch.get_default_dtype()
            loc_broadcast = loc_broadcast.to(dtype=dtype)
            scale_broadcast = scale_broadcast.to(dtype=dtype)

        if not torch.all(scale_broadcast > 0):
            raise ValueError("Expected scale > 0 elementwise.")

        super().__init__(batch_shape=loc_broadcast.shape, event_shape=())
        self.register_buffer("loc", loc_broadcast)
        self.register_buffer("scale", scale_broadcast)

    @property
    def params(self) -> tuple[Tensor, Tensor]:
        return self.loc, self.scale

    @property
    def mean(self) -> Tensor:
        return normal_mean(self.params)

    @property
    def median(self) -> Tensor:
        return normal_median(self.params)

    @property
    def mode(self) -> Tensor:
        return normal_mode(self.params)

    @property
    def variance(self) -> Tensor:
        return normal_variance(self.params)

    @property
    def stddev(self) -> Tensor:
        return normal_stddev(self.params)

    @override
    def sample(self, num: int = 1, /) -> Tensor:
        self.samples = normal_sample(self.params, num)
        return self.samples

    @override
    def log_prob(self, value: Tensor, /) -> Tensor:
        self.log_probs = normal_log_prob(self.params, value)
        return self.log_probs

    def cdf(self, value: Tensor, /) -> Tensor:
        return normal_cdf(self.params, value)

    def icdf(self, value: Tensor, /) -> Tensor:
        return normal_icdf(self.params, value)

    def entropy(self) -> Tensor:
        return normal_entropy(self.params)
