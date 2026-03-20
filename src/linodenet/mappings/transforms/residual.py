r"""ContractiveFlow implementation (iResNet-block)."""

__all__ = [
    "ResidualContraction",
    "ReZeroContraction",
    "ResidualContractionFallback",
]

import warnings
from typing import Final, Literal

import torch
from torch import Tensor, nn

from linodenet.mappings.base import TransformBase
from linodenet.mappings.bijections import SmoothSoftsign, TanhMap
from linodenet_special import fixpoint_solve


class ResidualContraction(TransformBase):
    r"""A residual flow based on a contraction layer.

    Forward: y ← x + g(x)
    Inverse: via fix-point iteration with implicit differentiation.
    """

    maxiter: Final[int]
    atol: Final[float]
    rtol: Final[float]

    def __init__(
        self,
        contraction: nn.Module,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.contraction = contraction
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol

    def encode(self, x: Tensor) -> Tensor:
        return x + self.contraction(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration."""
        # note: solve x = y - g(x) = f(x, y)
        return fixpoint_solve(
            lambda x: y - self.contraction(x),  # type: ignore[misc]
            y.clone(),
            maxiter=self.maxiter,
            atol=self.atol,
            rtol=self.rtol,
        )

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class ReZeroContraction(TransformBase):
    r"""A residual flow based on a scaled contraction layer.

    Forward: $y ← x + φ(ε)⋅g(x)$ with $φ(ε) ∈ (-1, 1)$.
    Inverse: via fixed point iteration with implicit differentiation.
    """

    scalar: Tensor
    r"""The unconstrained learnable scalar."""

    def __init__(
        self,
        contraction: nn.Module,
        *,
        scalar_map: nn.Module | Literal["smooth-softsign", "tanh"] = "smooth-softsign",
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        super().__init__()
        initial_value = torch.as_tensor(0.0)
        self.scalar = nn.Parameter(initial_value, requires_grad=True)
        self.scalar_map: nn.Module
        match scalar_map:
            case "smooth-softsign":
                self.scalar_map = SmoothSoftsign()
            case "tanh":
                self.scalar_map = TanhMap()
            case _:
                self.scalar_map = scalar_map
        self.contraction = contraction
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol

    def encode(self, x: Tensor) -> Tensor:
        alpha = self.scalar_map(self.scalar)
        return x + alpha * self.contraction(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration."""
        alpha = self.scalar_map(self.scalar)
        return fixpoint_solve(
            lambda x: y - alpha * self.contraction(x),  # type: ignore[misc]
            y.clone(),
            maxiter=self.maxiter,
            atol=self.atol,
            rtol=self.rtol,
        )

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class ResidualContractionFallback(TransformBase):
    r"""A residual flow based on a contraction layer.

    Forward: y ← x + g(x)
    Inverse: via fix-point iteration.

    The jacobian determinant of the forward transformation is:

    .. math:: \log\det(∂y/∂x) = \log\det(𝕀 + ∂g/∂x) = \tr\log(𝕀 + ∂g/∂x)

    Using the power series, this is

    .. math:: ∑_k (-1)ᵏ⁺¹ \tr((∂g/∂x)ᵏ)/k

    References:
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | https://proceedings.mlr.press/v97/behrmann19a.html
        - https://github.com/jhjacobsen/invertible-resnet
    """

    def __init__(
        self,
        contraction: nn.Module,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.contraction = contraction
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol

    def encode(self, x: Tensor) -> Tensor:
        return x + self.contraction(x)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or the elementwise tolerance threshold
        $|x'-x| ≤ \text{rtol}⋅|x| + \text{atol}$ is reached.
        """
        x = y.clone()

        for _ in range(self.maxiter):
            x_prev = x
            x = y - self.contraction(x_prev)
            residual = torch.abs(x - x_prev)
            tolerance = self.rtol * torch.abs(x) + self.atol

            if torch.all(residual <= tolerance):
                return x

        warnings.warn(
            f"No convergence in {self.maxiter} iterations. ",
            stacklevel=2,
        )
        return x

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
