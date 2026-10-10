r"""Implementation of the univariate Normal distribution."""

__all__ = [
    # Classes
    "Normal",
    "NormalParams",
]

import math
from typing import Final, NamedTuple, override

import torch
from torch import Tensor

from .base import DistributionBase

_SQRT_2: Final[float] = math.sqrt(2.0)
_HALF_LOG_2PI: Final[float] = 0.5 * math.log(2.0 * math.pi)
_HALF_LOG_2PIE: Final[float] = 0.5 * math.log(2.0 * math.pi * math.e)


class NormalParams(NamedTuple):
    r"""Normal parameters and operations for location `loc` and scale `scale`.

    Callers must provide floating-point parameters with the same shape, dtype,
    and device, satisfying `scale > 0` elementwise. Construction does not
    validate or broadcast parameters; use `Normal` for that. The tuple is
    immutable, but its tensors remain mutable.
    """

    loc: Tensor
    r"""Mean of the distribution."""
    scale: Tensor
    r"""Standard deviation of the distribution."""

    @property
    def mean(self) -> Tensor:
        r"""Mean of the distribution."""
        return self.loc

    @property
    def median(self) -> Tensor:
        r"""Median of the distribution."""
        return self.mean

    @property
    def mode(self) -> Tensor:
        r"""Mode of the distribution."""
        return self.mean

    @property
    def variance(self) -> Tensor:
        r"""Variance of the distribution."""
        return self.scale.square()

    @property
    def stddev(self) -> Tensor:
        r"""Standard deviation of the distribution."""
        return self.scale

    def sample(self, num: int = 1, /) -> Tensor:
        r"""Draw samples with shape `(num, *loc.shape)`."""
        shape = (num, *self.loc.shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value: Tensor, /) -> Tensor:
        r"""Log density at the given value."""
        z = (value - self.loc) / self.scale
        return -0.5 * z.square() - self.scale.log() - _HALF_LOG_2PI

    def cdf(self, value: Tensor, /) -> Tensor:
        r"""Cumulative distribution function."""
        z = (value - self.loc) / (self.scale * _SQRT_2)
        return 0.5 * (1.0 + torch.erf(z))

    def icdf(self, value: Tensor, /) -> Tensor:
        r"""Quantile function; callers must provide probabilities in `[0, 1]`."""
        return self.loc + self.scale * torch.erfinv(2.0 * value - 1.0) * _SQRT_2

    def entropy(self) -> Tensor:
        r"""Differential entropy of the distribution."""
        return self.scale.log() + _HALF_LOG_2PIE


class Normal(DistributionBase):
    r"""Univariate normal distribution parameterized by `loc` and `scale`.

    The mean `loc` and standard deviation `scale` are broadcast to a common batch
    shape. The scale must be positive elementwise. Samples are scalar events.
    """

    Params: Final = NormalParams
    r"""Parameter type for direct construction via `Normal.Params(loc, scale)`."""

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
    def params(self) -> NormalParams:
        r"""Parameters referencing the current registered location and scale."""
        return NormalParams(self.loc, self.scale)

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
