r"""Protocols and types for bijective transformations."""

__all__ = [
    "Bijection",
    "Transform",
]

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class Bijection[X, Y](Protocol):
    r"""Protocol for invertible layers."""

    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...


@runtime_checkable
class Transform[X, Y](Protocol):
    r"""Protocol for diffeomorphism."""

    @abstractmethod
    def encode_and_logabsdet(self, x: X, /) -> tuple[Y, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(self, y: Y, /) -> tuple[X, Tensor]: ...
