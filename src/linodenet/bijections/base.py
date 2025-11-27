r"""Protocols and base classes for bijections and transforms.

Note that `torch.distributions.Transform` has some differences:

- It is not a protocol.
- `log_abs_det_jacobian` always requires 2 arguments, $x$ and $y$.
  The rationale is that for certain bijections, the Jacobian is
  much faster to compute if the output is known.
  For example: if $f(x) = xᵃ$, then $\log\abs{\det 𝐃f[x]} = \log\abs{a⋅y/x}$.
  is more efficient than $\log\abs{a⋅xᵃ⁻¹}$.
  However, for many bijections, this is not true and knowing $y$ is not that helpful.
- Instead, it makes more sense to have 2 methods: `log_abs_det_jacobian(x)`
  and `value_and_log_abs_det_jacobian(x) -> tuple[Tensor, Tensor]`,
  similar to jax's `value_and_grad`.
  Alternatively, one can store `y` in a buffer and reuse it if needed, i.e.
  methods that need `y` can call:

>>> def log_abs_det_jacobian(self, x: Tensor, /, y: None | Tensor = None) -> Tensor:
>>>     if y is None:
>>>         if id(x) == id(self._last_x):
>>>             y = self._last_y
>>>         else:
>>>             y = self.forward(x)
"""

__all__ = [
    # Protocols & ABCs
    "BijectionBase",
    "TransformABC",
    "Bijection",
    "Transform",
    # Classes
    "BijectionSequence",
    "TransformSequence",
]

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn

from linodenet.containers import ModuleSequence


@runtime_checkable
class Bijection[X, Y](Protocol):
    r"""Protocol for invertible layers."""

    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...


@runtime_checkable
class Transform[X, Y](Bijection, Protocol):
    r"""Protocol for diffeomorphism with logabsdet."""

    @abstractmethod
    def encode_and_logabsdet(self, x: X, /) -> tuple[Y, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(self, y: Y, /) -> tuple[X, Tensor]: ...


class BijectionBase[X, Y](nn.Module, Bijection[X, Y]):
    r"""Abstract base class for invertible layers."""

    @abstractmethod
    def __invert__(self) -> Bijection[Y, X]: ...

    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...

    @jit.export
    def transform(self, x: X) -> Y:
        r"""Alias for encode."""
        return self.encode(x)

    @jit.export
    def inverse_transform(self, y: Y) -> X:
        r"""Alias for decode."""
        return self.decode(y)


class TransformABC[X, Y](nn.Module, Transform[X, Y]):
    r"""Abstract base class for diffeomorphism."""

    @abstractmethod
    def encode_and_logabsdet(self, x: X, /) -> tuple[Y, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(self, y: Y, /) -> tuple[X, Tensor]: ...

    @jit.export
    def transform_and_log_det(self, x: X) -> tuple[Y, Tensor]:
        r"""Alias for encode_and_logabsdet."""
        return self.encode_and_logabsdet(x)

    @jit.export
    def inverse_and_log_det(self, y: Y) -> tuple[X, Tensor]:
        r"""Alias for decode_and_logabsdet."""
        return self.decode_and_logabsdet(y)


class BijectionSequence(ModuleSequence[BijectionBase]):
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
