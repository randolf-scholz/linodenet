r"""Base classes for mappings."""

__all__ = [
    "Surjection",
    "SurjectionBase",
    "Embedding",
    "EmbeddingBase",
    "Projection",
    "ProjectionBase",
]


from abc import abstractmethod
from typing import Protocol, final, runtime_checkable

from torch import Tensor, jit, nn

from signatures import signature


@runtime_checkable
class Embedding[X, Y](Protocol):
    r"""Protocol for Embedding Components."""

    @abstractmethod
    @signature("(...) -> (...)")
    def forward(self, x: X, /) -> Y: ...

    @abstractmethod
    @signature("(...) -> (...)")
    def left_inverse(self, y: Y, /) -> X: ...


@runtime_checkable
class Surjection[X, Y](Protocol):
    r"""A protocol for surjections."""

    @abstractmethod
    def forward(self, x: X, /) -> Y: ...
    @abstractmethod
    def right_inverse(self, y: Y, /) -> X: ...


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

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Alias for `left_inverse` method."""
        return self.left_inverse(y)


class SurjectionBase(nn.Module, Surjection[Tensor, Tensor]):
    r"""Abstract Base Class for Surjection components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def right_inverse(self, y: Tensor, /) -> Tensor: ...

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Alias for `right_inverse` method."""
        return self.right_inverse(y)


class ProjectionBase(nn.Module):
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
    @jit.export
    @signature("(..., *ys) -> (..., *xs)")
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""Right inverse of the projection, i.e. the identity on the image.

        Args:
            y: The projected tensor.

        Returns:
            The input tensor as-is.
        """
        return y

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Alias for `right_inverse` method."""
        return self.right_inverse(y)
