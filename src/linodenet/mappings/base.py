r"""Base classes for mappings."""

__all__ = [
    # Protocols
    "Bijection",
    "Embedding",
    "Projection",
    "Surjection",
    # Classes
    "BijectionBase",
    "BijectionSequence",
    "EmbeddingBase",
    "InverseBijection",
    "ProjectionBase",
    "SurjectionBase",
]


from abc import abstractmethod
from collections.abc import Iterable
from typing import Protocol, final, runtime_checkable

from torch import Tensor, nn

from linodenet.nn import ModuleSequence
from signatures import signature


@runtime_checkable
class Embedding[X, Y](Protocol):
    r"""Protocol for Embedding Components."""

    @abstractmethod
    @signature("(...) -> (...)")
    def __call__(self, x: X, /) -> Y: ...

    @abstractmethod
    @signature("(...) -> (...)")
    def left_inverse(self, y: Y, /) -> X: ...


@runtime_checkable
class Surjection[X, Y](Protocol):
    r"""A protocol for surjections."""

    @abstractmethod
    def __call__(self, x: X, /) -> Y: ...
    @abstractmethod
    def right_inverse(self, y: Y, /) -> X: ...


@runtime_checkable
class Bijection[X, Y](Surjection[X, Y], Embedding[X, Y], Protocol):
    r"""Protocol for invertible layers."""

    @abstractmethod
    def __call__(self, x: X, /) -> Y: ...
    @abstractmethod
    def inverse(self, y: Y, /) -> X: ...

    def right_inverse(self, y: Y, /) -> X:
        return self.inverse(y)

    def left_inverse(self, y: Y, /) -> X:
        return self.inverse(y)


@runtime_checkable
class Projection[T](Surjection[T, T], Protocol):
    r"""Protocol for projections.

    Projections are a stronger form of surjections: we additionally require

    - The domain is a subset of the codomain
    -`right_inverse` is the identity map.

    That is, a projection is a mapping $φ:X→X$ such that $φ∘φ=φ$. In particular,
    $φ=i∘π$ for the embedding $i:\Im(φ)→X$ where $π:X→\Im(φ)$ is $φ$ viewed as a surjection onto its image.
    Then the identity map on the image of $φ$ is the right inverse of $π$.

    References:
        - https://en.wikipedia.org/wiki/Projection_(mathematics)
        - https://en.wikipedia.org/wiki/Projection_(linear_algebra)
    """

    @abstractmethod
    @signature("(..., *xs) -> (..., *ys)")
    def forward(self, x: T, /) -> T: ...

    @signature("(..., *ys) -> (..., *xs)")
    def right_inverse(self, y: T, /) -> T:
        r"""Right inverse of the projection, i.e. the identity on the image."""
        return y


class EmbeddingBase(nn.Module, Embedding[Tensor, Tensor]):
    r"""Abstract Base Class for Embedding components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def left_inverse(self, y: Tensor, /) -> Tensor: ...

    def encode(self, x: Tensor) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    def decode(self, y: Tensor) -> Tensor:
        r"""Alias for `left_inverse` method."""
        return self.left_inverse(y)


class SurjectionBase(nn.Module, Surjection[Tensor, Tensor]):
    r"""Abstract Base Class for Surjection components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def right_inverse(self, y: Tensor, /) -> Tensor: ...

    def encode(self, x: Tensor) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    def decode(self, y: Tensor) -> Tensor:
        r"""Alias for `right_inverse` method."""
        return self.right_inverse(y)


class ProjectionBase(SurjectionBase, Projection[Tensor]):
    r"""Abstract Base Class for Projection components."""

    @abstractmethod
    @signature("(..., *xs) -> (..., *ys)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the projection.

        Args:
            x: The input tensor to be projected.

        Returns:
            y: The projected tensor.
        """

    @final
    @signature("(..., *ys) -> (..., *xs)")
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""Right inverse of the projection, i.e. the identity on the image.

        Args:
            y: The projected tensor.

        Returns:
            The input tensor as-is.
        """
        return y

    def encode(self, x: Tensor) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    def decode(self, y: Tensor) -> Tensor:
        r"""Alias for `right_inverse` method."""
        return self.right_inverse(y)


class BijectionBase(nn.Module, Bijection[Tensor, Tensor]):
    r"""Base class for bijections operating on single tensor."""

    def __invert__(self) -> BijectionBase:
        return InverseBijection(self)

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def inverse(self, y: Tensor, /) -> Tensor: ...

    def encode(self, x: Tensor, /) -> Tensor:
        return self.forward(x)

    def decode(self, y: Tensor, /) -> Tensor:
        return self.inverse(y)


class InverseBijection[B: BijectionBase](BijectionBase):
    r"""Inverse of a bijection."""

    bijection: B
    r"""The bijection to be inverted."""

    def __init__(self, bijection: B) -> None:
        super().__init__()
        self.bijection = bijection

    def forward(self, x: Tensor, /) -> Tensor:
        return self.bijection.inverse(x)

    def inverse(self, y: Tensor, /) -> Tensor:
        return self.bijection.forward(y)


class BijectionSequence[B: BijectionBase](BijectionBase, ModuleSequence[B]):
    r"""Apply multiple bijections sequentially."""

    # noinspection PyMissingConstructor
    def __init__(self, modules: Iterable[B] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[B].__init__(self, modules)

    def __invert__(self) -> BijectionSequence:
        if type(self) is not BijectionSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )
        return BijectionSequence(~layer for layer in reversed(self))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self:
            x = layer.forward(x)
        return x

    def inverse(self, y: Tensor) -> Tensor:
        for layer in reversed(self):
            y = layer.inverse(y)
        return y
