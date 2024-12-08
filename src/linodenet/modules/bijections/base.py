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
    "Bijection",
    "BijectionABC",
    # Classes
    "iSequential",
]

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from torch import Tensor, jit, nn


@runtime_checkable
class Bijection[U, V](Protocol):
    r"""Protocol for invertible layers."""

    @abstractmethod
    def inverse(self) -> "Bijection[V, U]": ...
    @abstractmethod
    def encode(self, u: U, /) -> V: ...
    @abstractmethod
    def decode(self, v: V, /) -> U: ...

    def transform(self, u: U, /) -> V:
        r"""Alias for encode."""
        return self.encode(u)

    def inverse_transform(self, v: V, /) -> U:
        r"""Alias for decode."""
        return self.decode(v)


class BijectionABC[U, V](nn.Module, Bijection[U, V]):
    r"""Abstract base class for invertible layers."""

    @abstractmethod
    def encode(self, u: U, /) -> V: ...
    @abstractmethod
    def decode(self, v: V, /) -> U: ...


class iSequential(nn.Sequential):
    r"""Invertible Sequential model."""

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        return self(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        for layer in self[::-1]:  # traverse in reverse
            y = layer.decode(y)
        return y
