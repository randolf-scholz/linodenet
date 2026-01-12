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
    "Bijection",
    "Transform",
    # Classes
    "BijectionBase",
    "TransformBase",
    "BijectionSequence",
    "TransformSequence",
    "InverseBijection",
    "InverseTransform",
]

from abc import abstractmethod
from collections.abc import Iterable
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
class Transform[X, Y](Bijection[X, Y], Protocol):
    r"""Protocol for diffeomorphism with logabsdet."""

    @abstractmethod
    def encode_and_logabsdet(self, x: X, /) -> tuple[Y, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(self, y: Y, /) -> tuple[X, Tensor]: ...


class BijectionBase(nn.Module, Bijection[Tensor, Tensor]):
    r"""Base class for bijections operating on single tensor."""

    def __invert__(self) -> "BijectionBase":
        return InverseBijection(self)

    @abstractmethod
    def encode(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def decode(self, y: Tensor, /) -> Tensor: ...


class TransformBase(BijectionBase, Transform[Tensor, Tensor]):
    r"""Base class for transforms operating on single tensor."""

    def __invert__(self) -> "TransformBase":
        return InverseTransform(self)

    @abstractmethod
    def encode(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def decode(self, y: Tensor, /) -> Tensor: ...

    @abstractmethod
    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]: ...


class InverseBijection[B: BijectionBase](BijectionBase):
    r"""Inverse of a bijection."""

    bijection: B
    r"""The bijection to be inverted."""

    def __init__(self, bijection: B) -> None:
        super().__init__()
        self.bijection = bijection

    @jit.export
    def encode(self, x: Tensor, /) -> Tensor:
        return self.bijection.decode(x)

    @jit.export
    def decode(self, y: Tensor, /) -> Tensor:
        return self.bijection.encode(y)


class InverseTransform[T: TransformBase](TransformBase):
    r"""Inverse of a transform."""

    transform: T
    r"""The transform to be inverted."""

    def __init__(self, transform: T) -> None:
        super().__init__()
        self.transform = transform

    @jit.export
    def encode(self, x: Tensor, /) -> Tensor:
        return self.transform.decode(x)

    @jit.export
    def decode(self, y: Tensor, /) -> Tensor:
        return self.transform.encode(y)

    @jit.export
    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y, logabsdet = self.transform.decode_and_logabsdet(x)
        return y, -logabsdet

    @jit.export
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x, logabsdet = self.transform.encode_and_logabsdet(y)
        return x, -logabsdet


class BijectionSequence[B: BijectionBase](BijectionBase, ModuleSequence[B]):
    r"""Apply multiple bijections sequentially."""

    def __init__(self, modules: Iterable[B] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[B].__init__(self, modules)

    def __invert__(self) -> "BijectionSequence":
        if type(self) is not BijectionSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )
        return BijectionSequence([~layer for layer in self[::-1]])

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


class TransformSequence[T: TransformBase](BijectionSequence[T]):
    r"""Apply multiple transforms sequentially."""

    def __invert__(self) -> "TransformSequence":
        return TransformSequence([~layer for layer in self[::-1]])

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

    @jit.export
    def encode_and_logabsdet(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in self:
            x, logabsdet = layer.encode_and_logabsdet(x)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return x, logabsdet

    @jit.export
    def decode_and_logabsdet(self, y: Tensor) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in self[::-1]:  # traverse in reverse
            y, logabsdet = layer.decode_and_logabsdet(y)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return y, logabsdet
