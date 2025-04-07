r"""Implementation of invertible layers.

Layers:

- Affine: $y = Ax+b$ and $x = A⁻¹(y-b)$
    - A diagonal
    - A triangular
    - A tridiagonal
- Element-wise:
    - Monotonic
- Shears (coupling flows): $y_A = f(x_A, x_B)$ and $y_B = x_B$
    - Example: $y_A = x_A + e^{x_B}$ and $y_B = x_B$
- Residual: $y = x + F(x)$
    - Contractive: $y = F(x)$ with $‖F‖<1$
    - Low Rank Perturbation: $y = x + ABx$
- Continuous Time Flows: $ẋ=f(t, x)$
"""

__all__ = [
    # Protocols & ABCs
    "BijectionABC",
    "TransformABC",
    # Classes
    "BijectionSequence",
    "TransformSequence",
]

from abc import abstractmethod

import torch
from torch import Tensor, jit, nn

from linodenet.modules.bijections.abstract import Bijection, Transform
from linodenet.torch_generics import ModuleSequence


class BijectionABC[X, Y](nn.Module, Bijection[X, Y]):
    r"""Abstract base class for invertible layers."""

    @abstractmethod
    def __invert__(self) -> "Bijection[Y, X]": ...

    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...

    def transform(self, x: X, /) -> Y:
        r"""Alias for encode."""
        return self.encode(x)

    def inverse_transform(self, y: Y, /) -> X:
        r"""Alias for decode."""
        return self.decode(y)


class TransformABC[X, Y](nn.Module, Transform[X, Y]):
    r"""Abstract base class for diffeomorphism."""

    @abstractmethod
    def encode_and_logabsdet(self, x: X, /) -> tuple[Y, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(self, y: Y, /) -> tuple[X, Tensor]: ...

    # def log_abs_det_jacobian(self, x: X, y: Y, /) -> Tensor: ...
    # NOTE: By inverse function theorem, `log|det Df⁻¹(y)| = log|det Df(x)⁻¹| = log|1/det Df(x)| = -log|det Df(x)|`


class BijectionSequence(ModuleSequence[BijectionABC]):
    r"""Invertible Sequential model."""

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        for layer in self:
            x = layer.encode(x)
        return x

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        for layer in self[::-1]:  # traverse in reverse
            y = layer.decode(y)
        return y


class TransformSequence[T](ModuleSequence[TransformABC[T, T]]):
    r"""Sequence of transformations."""

    @jit.export
    def encode_and_logabsdet(self, x: T) -> tuple[T, Tensor]:
        logabsdets: list[Tensor] = []
        for layer in self:
            x, logabsdet = layer.encode_and_logabsdet(x)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return x, logabsdet

    @jit.export
    def decode_and_logabsdet(self, y: T) -> tuple[T, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in self[::-1]:  # traverse in reverse
            y, logabsdet = layer.decode_and_logabsdet(y)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return y, logabsdet
