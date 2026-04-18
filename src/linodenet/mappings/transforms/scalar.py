r"""Scalar Transforms."""

__all__ = [
    "CELU",
    "ELU",
    "EntLU",
    "Sigmoid",
    "SmoothSoftsign",
    "Softplus",
    "Softsign",
    "Tanh",
    "Tanhshrink",
]


from typing import Final

import torch
from torch import Tensor
from torch.nn import functional as F

from linodenet.domains import ScalarDomain, ScalarDomains
from signatures import signature

from .base import TransformBase


class Sigmoid(TransformBase):
    r"""Map tensor entries elementwise via $x ↦ \sigma(x) = 1/(1 + \exp(-x))$.

    The inverse is $y ↦ \sigma^{-1}(y) = \log(y/(1-y))$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.OPEN_UNIT_BALL

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return torch.sigmoid(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.log(y / (1 - y))


class Tanh(TransformBase):
    r"""Maps tensor elementwise via via $x ↦ \tanh(x)$."""

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.OPEN_UNIT_BALL

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return torch.tanh(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.atanh(y)


class Softsign(TransformBase):
    r"""Maps tensor elementwise via $x ↦ x/(1 + |x|)$.

    The inverse is $y ↦ y/(1 - |y|)$.
    """

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.softsign(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return y / (1 - y.abs())


class SmoothSoftsign(TransformBase):
    r"""Maps tensor elementwise via $x ↦ 2x/(1 + √(1 + 4x²))$."""

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.OPEN_UNIT_BALL

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return 2 * x / (1 + torch.sqrt(1 + 4 * x.square()))

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return y / (1 - y.square())


class Softplus(TransformBase):
    r"""Maps tensor elementwise via $x ↦ \log(1 + \exp(x))$.

    The inverse is $y ↦ \log(\exp(y) - 1)$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.POSITIVE_REALS

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.softplus(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.where(y > 20, y, y + torch.log(-torch.expm1(-y)))


class ELU(TransformBase):
    r"""Maps tensor elementwise via $x ↦ ⟦x > 0 ? x : α(\exp(x) -1)⟧$.

    The inverse is y ↦ ⟦y > 0 ? y : α \log(1 + y/α)⟧.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    alpha: Final[float]

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be a positive float.")
        self.alpha = alpha

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.elu(x, self.alpha)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.where(y > 0, y, self.alpha * torch.log1p(y / self.alpha))


class CELU(TransformBase):
    r"""Maps tensor elementwise via $x ↦ ⟦x > 0 ? x : α(\exp(x/α) -1)⟧$.

    The inverse is: $y ↦ ⟦y > 0 ? y : α \log(1 + y/α)⟧$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.Interval("(-1, ∞)")

    alpha: Final[float]

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be a positive float.")
        self.alpha = alpha

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.celu(x, self.alpha)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.where(y > 0, y, self.alpha * torch.log1p(y / self.alpha))


class EntLU(TransformBase):
    r"""Maps tensor elementwise via $x ↦ ⟦x>0 \? eᴴ⁽¹⁻ˣ⁾ : x + 1⟧$.

    The inverse is $y ↦ ⟦y < 1 \? 1 - \log(1/y)/W(\log(1/y)) : y-1⟧$.

    Here, $H(x) = -x\log(x)$ is the entropy function. (EntLU = Entropic Linear Unit)
    """

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return torch.where(x > 0, x + 1, torch.exp(torch.special.entr(x)))

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/108948")


class Tanhshrink(TransformBase):
    r"""Maps tensor elementwise via $x ↦ x - tanh(x)$.

    The inverse is $y ↦ ½ W(2y+2, 2y-2, -1)$, where $W$ is the generalized
    Lambert-W function, that is $W(⋅, t, s)$ is the inverse of $u ↦ eᵘ(u-t)/(u-s)$.
    """

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.tanhshrink(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/108948")
