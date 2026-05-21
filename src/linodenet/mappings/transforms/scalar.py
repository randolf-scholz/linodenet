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


import math
from typing import Final

import torch
from torch import Tensor
from torch.nn import functional as F

from linodenet.domains import ScalarDomain, ScalarDomains
from linodenet.mappings.base import TransformBase
from signatures import signature


class Sigmoid(TransformBase):
    r"""Map tensor entries elementwise via $x ↦ \sigma(x) = 1/(1 + \exp(-x))$.

    The inverse is $y ↦ \log(y/(1-y))$.

    The derivative is: $\sigma'(x) = \sigma(x) (1-\sigma(x))$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.OPEN_UNIT_BALL

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return torch.sigmoid(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.log(y / (1 - y))

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        logabsdet = F.logsigmoid(x) + F.logsigmoid(-x)
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        logabsdet = -torch.log(y) - torch.log1p(-y)
        return x, logabsdet


class Tanh(TransformBase):
    r"""Maps tensor elementwise via via $x ↦ \tanh(x)$.

    The inverse is $y ↦ \atanh(y) = ½ \log((1+y)/(1-y))$.

    The derivative is: $\frac{d}{dx}\tanh(x) = 1-\tanh²(x)$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.OPEN_UNIT_BALL

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return torch.tanh(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.atanh(y)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        # d/dx tanh(x) = 1/cosh(x)²;
        # log(cosh(x)) = log(0.5) + log(e⁻ˣ + eˣ)
        # log(1/cosh(x)²) = -2 log(cosh(x)) = -2 (log(0.5) + log(e⁻ˣ + eˣ))
        y = self.forward(x)
        logabsdet = -2.0 * (math.log(0.5) + torch.logaddexp(x, -x))
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        logabsdet = -torch.log1p(-y.square())
        return x, logabsdet


class Softsign(TransformBase):
    r"""Maps tensor elementwise via $x ↦ x/(1 + |x|)$.

    The inverse is $y ↦ y/(1 - |y|)$.

    The derivative is: $\frac{d}{dx}\frac{x}{1+|x|} = \frac{1}{(1+|x|)²}$.
    """

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.softsign(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return y / (1 - y.abs())

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        logabsdet = -2 * torch.log1p(x.abs())
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        logabsdet = -2 * torch.log1p(-y.abs())
        return x, logabsdet


class SmoothSoftsign(TransformBase):
    r"""Maps tensor elementwise via $x ↦ 2x/(1 + √(1 + 4x²))$.

    The inverse is $y ↦ y/(1 - y²)$.

    The derivative is: $\frac{d}{dx}\frac{2x}{1 + √(1 + 4x²)}
    = \frac{2}{√(1+4x²)(1 + √(1 + 4x²))}$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.OPEN_UNIT_BALL

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return 2 * x / (1 + torch.sqrt(1 + 4 * x.square()))

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return y / (1 - y.square())

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        root = torch.sqrt(1 + 4 * x.square())
        logabsdet = math.log(2.0) - torch.log(root) - torch.log1p(root)
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        logabsdet = torch.log1p(y.square()) - 2 * torch.log1p(-y.square())
        return x, logabsdet


class Softplus(TransformBase):
    r"""Maps tensor elementwise via $x ↦ \log(1 + \exp(x))$.

    The inverse is $y ↦ \log(\exp(y) - 1)$.
    The derivative is: $\frac{d}{dx}\log(1+\exp(x)) = \sigma(x)$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.POSITIVE_REALS

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.softplus(x)

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return torch.where(y > 20, y, y + torch.log(-torch.expm1(-y)))

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        logabsdet = F.logsigmoid(x)
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        logabsdet = -torch.log(-torch.expm1(-y))
        return x, logabsdet


class ELU(TransformBase):
    r"""Maps tensor elementwise via $x ↦ ⟦x > 0 ? x : α(\exp(x) -1)⟧$.

    The inverse is y ↦ ⟦y > 0 ? y : \log(1 + y/α)⟧.

    The derivative is: ⟦x > 0 ? 1 : α\exp(x)⟧.
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
        return torch.where(y > 0, y, torch.log1p(y / self.alpha))

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        zeros = torch.zeros_like(x)
        log_alpha = math.log(self.alpha)
        logabsdet = torch.where(x > 0, zeros, x + log_alpha)
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        zeros = torch.zeros_like(y)
        logabsdet = torch.where(y > 0, zeros, -(y + self.alpha).log())
        return x, logabsdet


class CELU(TransformBase):
    r"""Maps tensor elementwise via $x ↦ ⟦x > 0 ? x : α(\exp(x/α) -1)⟧$.

    The inverse is: $y ↦ ⟦y > 0 ? y : α \log(1 + y/α)⟧$.

    The derivative is: ⟦x > 0 ? 1 : \exp(x/α)⟧.
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

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        zeros = torch.zeros_like(x)
        logabsdet = torch.where(x > 0, zeros, x / self.alpha)
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        logabsdet = torch.where(
            y > 0,
            torch.zeros_like(y),
            math.log(self.alpha) - (y + self.alpha).log(),
        )
        return x, logabsdet


class EntLU(TransformBase):
    r"""Maps tensor elementwise via $x ↦ ⟦x>0 ? x + 1 : eᴴ⁽¹⁻ˣ⁾⟧$.

    The inverse is $y ↦ ⟦y > 1 ? y-1 : 1 - \log(1/y)/W(\log(1/y))⟧$.

    The derivative is: ⟦x > 0 ? 1 : eᴴ⁽¹⁻ˣ⁾ (1 + \log(1-x))⟧.

    Here, $H(x)$ is PyTorch's entropy primitive. (EntLU = Entropic Linear Unit)
    """

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        # one_m_x avoids NaN production in entr, helps with gradients.
        one_m_x = torch.where(x > 0, torch.zeros_like(x), 1 - x)
        return torch.where(x > 0, x + 1, torch.exp(torch.special.entr(one_m_x)))

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        logabsdet = torch.where(
            x > 0,
            torch.zeros_like(x),
            y.log() + torch.log1p(torch.log1p(-x)),
        )
        return y, logabsdet

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/108948")

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/108948")


class Tanhshrink(TransformBase):
    r"""Maps tensor elementwise via $x ↦ x - tanh(x)$.

    The inverse is $y ↦ ½ W(2y+2, 2y-2, -1)$, where $W$ is the generalized
    Lambert-W function, that is $W(⋅, t, s)$ is the inverse of $u ↦ eᵘ(u-t)/(u-s)$.

    The derivative is: $\frac{d}{dx}(x-\tanh(x)) = \tanh²(x)$.
    """

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return F.tanhshrink(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        logabsdet = torch.tanh(x).square().log()
        return y, logabsdet

    @signature("(...) -> (...)")
    def decode(self, y: Tensor, /) -> Tensor:
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/108948")

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/108948")
