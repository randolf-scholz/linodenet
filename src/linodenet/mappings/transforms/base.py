r"""Base classes for transforms operating on single tensor."""

__all__ = [
    "Transform",
    "TransformBase",
    "InverseTransform",
    "TransformSequence",
]

from abc import abstractmethod
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from linodenet.mappings.base import Bijection, BijectionBase
from linodenet.nn import ModuleSequence


@runtime_checkable
class Transform[X, Y](Bijection[X, Y], Protocol):
    r"""Protocol for diffeomorphism with logabsdet."""

    @abstractmethod
    def encode_and_logabsdet(self, x: X, /) -> tuple[Y, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(self, y: Y, /) -> tuple[X, Tensor]: ...

    def encode(self, x: X, /) -> Y:
        y, _ = self.encode_and_logabsdet(x)
        return y

    def decode(self, y: Y, /) -> X:
        x, _ = self.decode_and_logabsdet(y)
        return x


class TransformBase(BijectionBase, Transform[Tensor, Tensor]):
    r"""Base class for transforms operating on single tensor."""

    def __invert__(self) -> TransformBase:
        return InverseTransform(self)

    @abstractmethod
    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]: ...

    def encode(self, x: Tensor, /) -> Tensor:
        y, _ = self.encode_and_logabsdet(x)
        return y

    def decode(self, y: Tensor, /) -> Tensor:
        x, _ = self.decode_and_logabsdet(y)
        return x

    def forward(self, x: Tensor, /) -> Tensor:
        return self.encode(x)

    def inverse(self, y: Tensor, /) -> Tensor:
        return self.decode(y)


class InverseTransform[T: TransformBase](TransformBase, ModuleSequence[T]):
    r"""Inverse of a transform."""

    transform: T
    r"""The transform to be inverted."""

    def __init__(self, transform: T) -> None:
        super().__init__()
        self.transform = transform

    def encode(self, x: Tensor, /) -> Tensor:
        return self.transform.decode(x)

    def decode(self, y: Tensor, /) -> Tensor:
        return self.transform.encode(y)

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y, logabsdet = self.transform.decode_and_logabsdet(x)
        return y, -logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x, logabsdet = self.transform.encode_and_logabsdet(y)
        return x, -logabsdet


class TransformSequence[T: TransformBase](TransformBase, ModuleSequence[T]):
    r"""Apply multiple transforms sequentially."""

    # noinspection PyMissingConstructor
    def __init__(self, modules: Iterable[T] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[T].__init__(self, modules)

    def __invert__(self) -> TransformSequence:
        if type(self) is not TransformSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )
        return TransformSequence(~layer for layer in reversed(self))

    def encode(self, x: Tensor) -> Tensor:
        for layer in self:
            x = layer.encode(x)
        return x

    def decode(self, y: Tensor) -> Tensor:
        for layer in reversed(self):
            y = layer.decode(y)
        return y

    def encode_and_logabsdet(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in self:
            x, logabsdet = layer.encode_and_logabsdet(x)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return x, logabsdet

    def decode_and_logabsdet(self, y: Tensor) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in reversed(self):
            y, logabsdet = layer.decode_and_logabsdet(y)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return y, logabsdet
