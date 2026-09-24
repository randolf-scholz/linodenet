r"""Parametrizations utilities."""

__all__ = [
    "ReZero",
    "Symmetric",
    "SkewSymmetric",
    "PositiveScalarMatrix",
    "PositiveDiagonalMatrix",
    "PositiveDefinite",
]

from typing import cast

import torch
from torch import Tensor, nn


class ReZero[
    M: nn.Module = nn.Module,
    S: nn.Module = nn.Module,
](nn.Module):
    r"""ReZero module, learnable scalar with optional transformation.

    .. math:: x ⟼ φ(ε) ⋅ f(x)
    """

    scalar: Tensor
    r"""PARAM: The scalar to multiply the inputs by."""
    scalar_map: S
    r"""MODULE: Map applied to the scalar before scaling the input."""
    module: M
    r"""MODULE: Map applied to the inputs before scaling them."""

    def __init__[U: nn.Module = nn.Identity, V: nn.Module = nn.Identity](
        self: ReZero[U, V],
        module: U | None = None,
        *,
        scalar_map: V | None = None,
        initial_value: Tensor | float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.scalar = nn.Parameter(
            torch.as_tensor(initial_value), requires_grad=learnable
        )
        self.module = cast("U", nn.Identity() if module is None else module)
        self.scalar_map = cast("V", nn.Identity() if scalar_map is None else scalar_map)

    def forward(self, x: Tensor, /) -> Tensor:
        return self.scalar_map(self.scalar) * self.module(x)

    def right_inverse(self, y: Tensor, /) -> Tensor | None:
        if (right_inverse := getattr(self.module, "right_inverse", None)) is None:
            return None

        assert callable(right_inverse)
        return right_inverse(y / self.scalar_map(self.scalar))  # type: ignore[return-value]


class Symmetric(nn.Module):
    r"""Symmetric parametrization of the kernel."""

    def forward(self, x: Tensor, /) -> Tensor:
        return (x + x.mT) / 2


class SkewSymmetric(nn.Module):
    r"""Skew-symmetric parametrization of the kernel."""

    def forward(self, x: Tensor, /) -> Tensor:
        return (x - x.mT) / 2


class PositiveDefinite(nn.Module):
    r"""Parametrize positive definite matrices via a lower-triangular matrix with log-diagonal."""

    def forward(self, tensor: Tensor, /) -> Tensor:
        L = (
            tensor.tril(diagonal=-1)
            + tensor.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
        )
        return L @ L.mT


class PositiveScalarMatrix(nn.Module):
    r"""Parametrization of a positive scalar matrix $eᶜ𝕀$."""

    eye: Tensor

    def __init__(self, size: int, log_scale: Tensor | float = 0.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.as_tensor(log_scale))
        self.register_buffer("eye", torch.eye(size))

    def forward(self) -> Tensor:
        return self.eye * torch.exp(self.log_scale)


class PositiveDiagonalMatrix(nn.Module):
    r"""Parametrization of a positive diagonal matrix as $\diag(eᵛ)$."""

    def __init__(self, size: int, log_scales: Tensor | float = 0.0) -> None:
        super().__init__()
        self.register_buffer("eye", torch.eye(size))
        # store diagonal (v_1, v_2, ..., v_n)
        self.log_scales = nn.Parameter(torch.as_tensor(log_scales).expand(size))

    def forward(self) -> Tensor:
        return self.log_scales.diag_embed()
