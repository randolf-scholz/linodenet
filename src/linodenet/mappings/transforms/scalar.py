r"""Scalar Transforms."""

__all__ = [
    "CELU",
    "ELU",
    "EntLU",
    "Sigmoid",
    "SmoothSoftsign",
    "Softplus",
    "Softsign",
    "Sinh",
    "Tanh",
    "Tanhshrink",
    "ConjugatedAffineFlow",
]


import math
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from linodenet.domains import ScalarDomain, ScalarDomains
from linodenet.mappings.abstract import Transform
from linodenet.nn.containers import Constant
from signatures import signature

_LOG2 = math.log(2.0)


class Sinh(nn.Module, Transform):
    r"""Maps tensor elementwise via $x ↦ \sinh(x) = (eˣ - e⁻ˣ)/2$.

    The inverse is $y ↦ \asinh(y) = \log(y + √(y² + 1))$.

    The derivative is: $\frac{d}{dx}\sinh(x) = \cosh(x)$.

    The logabsdet is: $\log\cosh(x) = ...$
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return x.sinh()

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return y.arcsinh()

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        return x.sinh(), x.cosh().log()

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        return y.arcsinh(), -0.5 * y.square().log1p()


class Sigmoid(nn.Module, Transform):
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


class Tanh(nn.Module, Transform):
    r"""Maps tensor elementwise via via $x ↦ \tanh(x)$.

    The inverse is $y ↦ \atanh(y) = ½ \log((1+y)/(1-y))$.

    The derivative is: $\frac{d}{dx}\tanh(x) = 1-\tanh²(x)$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.OPEN_UNIT_BALL

    @signature("(...) -> (...)")
    def forward(self, x: Tensor, /) -> Tensor:
        return x.tanh()

    @signature("(...) -> (...)")
    def inverse(self, y: Tensor, /) -> Tensor:
        return y.arctanh()

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        # d/dx tanh(x) = 1/cosh(x)²;
        # log(cosh(x)) = log(0.5) + log(e⁻ˣ + eˣ)
        # log(1/cosh(x)²) = -2 log(cosh(x)) = -2 (log(0.5) + log(e⁻ˣ + eˣ))
        y = self.forward(x)
        logabsdet = -2.0 * (-_LOG2 + torch.logaddexp(x, -x))
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        logabsdet = -torch.log1p(-y.square())
        return x, logabsdet


class Softsign(nn.Module, Transform):
    r"""Maps tensor elementwise via $x ↦ x/(1 + |x|)$.

    The inverse is $y ↦ y/(1 - |y|)$.

    The derivative is: $\frac{d}{dx}\frac{x}{1+|x|} = \frac{1}{(1+|x|)²}$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.OPEN_UNIT_BALL

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


class SmoothSoftsign(nn.Module, Transform):
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
        logabsdet = _LOG2 - torch.log(root) - torch.log1p(root)
        return y, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        logabsdet = torch.log1p(y.square()) - 2 * torch.log1p(-y.square())
        return x, logabsdet


class Softplus(nn.Module, Transform):
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


class ELU(nn.Module, Transform):
    r"""Maps tensor elementwise via $x ↦ ⟦x > 0 ? x : α(\exp(x) -1)⟧$.

    The inverse is y ↦ ⟦y > 0 ? y : \log(1 + y/α)⟧.

    The derivative is: ⟦x > 0 ? 1 : α\exp(x)⟧.
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


class CELU(nn.Module, Transform):
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


class EntLU(nn.Module, Transform):
    r"""Maps tensor elementwise via $x ↦ ⟦x>0 ? x + 1 : eᴴ⁽¹⁻ˣ⁾⟧$.

    The inverse is $y ↦ ⟦y > 1 ? y-1 : 1 - \log(1/y)/W(\log(1/y))⟧$.

    The derivative is: ⟦x > 0 ? 1 : eᴴ⁽¹⁻ˣ⁾ (1 + \log(1-x))⟧.

    Here, $H(x)$ is PyTorch's entropy primitive. (EntLU = Entropic Linear Unit)
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.POSITIVE_REALS

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


class Tanhshrink(nn.Module, Transform):
    r"""Maps tensor elementwise via $x ↦ x - tanh(x)$.

    The inverse is $y ↦ ½ W(2y+2, 2y-2, -1)$, where $W$ is the generalized
    Lambert-W function, that is $W(⋅, t, s)$ is the inverse of $u ↦ eᵘ(u-t)/(u-s)$.

    The derivative is: $\frac{d}{dx}(x-\tanh(x)) = \tanh²(x)$.
    """

    DOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE
    CODOMAIN: Final[ScalarDomain] = ScalarDomains.REAL_LINE

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


class ConjugatedAffineFlow(nn.Module, Transform):
    r"""Maps tensors elements-wise via $x ↦ ϕ⁻¹(α(ε)⋅ϕ(x)+β(ε))$.

    Here, $ϕ$ is a diffeomorphism, and both $α(0)=1$, $β(0)=0$ are smooth with $α(ε)≠0$.

    simple choices are: $α(z)=ℯᶜᶻ$, $β(z)=dz$ for some constants $c,d∈ℝ$.
    """

    def __init__(
        self,
        conjugate_map: Transform,
        *,
        alpha_map: nn.Module | None,
        beta_map: nn.Module | None,
    ) -> None:
        super().__init__()
        self.conjugate_map = conjugate_map
        self.parameter = nn.Parameter(torch.tensor(0.0))
        self.alpha_map = Constant(1.0) if alpha_map is None else alpha_map
        self.beta_map = nn.Identity() if beta_map is None else beta_map

    def encode(self, x: Tensor, /) -> Tensor:
        # y = ϕ⁻¹(α(ε)⋅ϕ(x)+β(ε))
        u = self.conjugate_map.encode(x)
        alpha = self.alpha_map(self.parameter)
        beta = self.beta_map(self.parameter)
        z = alpha * u + beta
        return self.conjugate_map.decode(z)

    def decode(self, y: Tensor, /) -> Tensor:
        # x = ϕ⁻¹( (ϕ(y) - β(ε))/α(ε))
        u = self.conjugate_map.encode(y)
        alpha = self.alpha_map(self.parameter)
        beta = self.beta_map(self.parameter)
        z = (u - beta) / alpha
        return self.conjugate_map.decode(z)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        # y = ϕ⁻¹(α(ε)⋅ϕ(x)+β(ε))
        u, ldj_enc = self.conjugate_map.encode_and_logabsdet(x)
        alpha = self.alpha_map(self.parameter)
        beta = self.beta_map(self.parameter)
        z = alpha * u + beta
        y, ldj_dec = self.conjugate_map.decode_and_logabsdet(z)
        return y, (ldj_enc + alpha.abs().log() + ldj_dec)

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        # x = ϕ⁻¹( (ϕ(y) - β(ε))/α(ε))
        u, ldj_enc = self.conjugate_map.encode_and_logabsdet(y)
        alpha = self.alpha_map(self.parameter)
        beta = self.beta_map(self.parameter)
        z = (u - beta) / alpha
        x, ldj_dec = self.conjugate_map.decode_and_logabsdet(z)
        return x, (ldj_enc - alpha.abs().log() + ldj_dec)
