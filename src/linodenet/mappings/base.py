r"""Base classes for mappings."""

__all__ = [
    # Protocols
    "Bijection",
    "Embedding",
    "Projection",
    "Surjection",
    "Transform",
    # Classes
    "BijectionBase",
    "BijectionSequence",
    "EmbeddingBase",
    "InverseBijection",
    "InverseTransform",
    "ProjectionBase",
    "SurjectionBase",
    "TransformBase",
    "TransformSequence",
]


from abc import abstractmethod
from collections.abc import Iterable
from typing import Protocol, final, runtime_checkable

import torch
from torch import Tensor, nn

from linodenet.nn import ModuleSequence
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
class Bijection[X, Y](Surjection[X, Y], Embedding[X, Y], Protocol):
    r"""Protocol for invertible layers."""

    @abstractmethod
    def forward(self, x: X, /) -> Y: ...
    @abstractmethod
    def inverse(self, y: Y, /) -> X: ...

    def right_inverse(self, y: Y, /) -> X:
        return self.inverse(y)

    def left_inverse(self, y: Y, /) -> X:
        return self.inverse(y)


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


class InverseTransform[T: TransformBase](TransformBase):
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


class TransformSequence[T: TransformBase](BijectionSequence[T]):
    r"""Apply multiple transforms sequentially."""

    def __invert__(self) -> TransformSequence:
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
