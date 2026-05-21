r"""Base classes for mappings."""

__all__ = [
    # Classes
    "BijectionBase",
    "EmbeddingBase",
    "ProjectionBase",
    "SurjectionBase",
    "TransformBase",
    # Conditional Base Classes
    "ConditionalBijectionBase",
    "ConditionalEmbeddingBase",
    "ConditionalProjectionBase",
    "ConditionalSurjectionBase",
    "ConditionalTransformBase",
    # inverse classes
    "InverseBijection",
    "InverseTransform",
    # inverse + conditional
    "InverseConditionalBijection",
    "InverseConditionalTransform",
    # sequence classes
    "BijectionSequence",
    "TransformSequence",
    # sequence + conditional
    "ConditionalBijectionSequence",
    "ConditionalTransformSequence",
]


from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, final

import torch
from torch import Tensor, nn

from linodenet.nn import ModuleSequence
from signatures import signature

from .abstract import (
    Bijection,
    ConditionalBijection,
    ConditionalEmbedding,
    ConditionalProjection,
    ConditionalSurjection,
    ConditionalTransform,
    Embedding,
    Projection,
    Surjection,
    Transform,
)


class EmbeddingBase(nn.Module, Embedding[Tensor, Tensor]):
    r"""Abstract Base Class for Embedding components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def left_inverse(self, y: Tensor, /) -> Tensor: ...

    def encode(self, x: Tensor, /) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    def decode(self, y: Tensor, /) -> Tensor:
        r"""Alias for `left_inverse` method."""
        return self.left_inverse(y)


class SurjectionBase(nn.Module, Surjection[Tensor, Tensor]):
    r"""Abstract Base Class for Surjection components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def right_inverse(self, y: Tensor, /) -> Tensor: ...

    def encode(self, x: Tensor, /) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    def decode(self, y: Tensor, /) -> Tensor:
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
    def right_inverse(self, y: Tensor, /) -> Tensor:
        r"""Right inverse of the projection, i.e. the identity on the image.

        Args:
            y: The projected tensor.

        Returns:
            The input tensor as-is.
        """
        return y

    def encode(self, x: Tensor, /) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x)

    def decode(self, y: Tensor, /) -> Tensor:
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


class ConditionalEmbeddingBase[Z](
    nn.Module,
    ConditionalEmbedding[Tensor, Tensor, Z],
):
    r"""Abstract base class for conditional embedding components."""

    @abstractmethod
    def forward(self, x: Tensor, context: Z, /) -> Tensor: ...
    @abstractmethod
    def left_inverse(self, y: Tensor, context: Z, /) -> Tensor: ...

    def encode(self, x: Tensor, context: Z, /) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x, context)

    def decode(self, y: Tensor, context: Z, /) -> Tensor:
        r"""Alias for `left_inverse` method."""
        return self.left_inverse(y, context)


class ConditionalSurjectionBase[Z](
    nn.Module,
    ConditionalSurjection[Tensor, Tensor, Z],
):
    r"""Abstract base class for conditional surjection components."""

    @abstractmethod
    def forward(self, x: Tensor, context: Z, /) -> Tensor: ...
    @abstractmethod
    def right_inverse(self, y: Tensor, context: Z, /) -> Tensor: ...

    def encode(self, x: Tensor, context: Z, /) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x, context)

    def decode(self, y: Tensor, context: Z, /) -> Tensor:
        r"""Alias for `right_inverse` method."""
        return self.right_inverse(y, context)


class ConditionalProjectionBase[Z](
    ConditionalSurjectionBase[Z],
    ConditionalProjection[Tensor, Z],
):
    r"""Abstract base class for conditional projection components."""

    @abstractmethod
    @signature("(..., *xs) -> (..., *ys)")
    def forward(self, x: Tensor, context: Z, /) -> Tensor:
        r"""Forward pass of the conditional projection.

        Args:
            x: The input tensor to be projected.
            context: The conditioning information.

        Returns:
            y: The projected tensor.
        """

    @final
    @signature("(..., *ys) -> (..., *xs)")
    def right_inverse(self, y: Tensor, context: Z, /) -> Tensor:  # noqa: ARG002
        r"""Right inverse of the conditional projection for a fixed context.

        Args:
            y: The projected tensor.
            context: The conditioning information.

        Returns:
            The input tensor as-is.
        """
        return y

    def encode(self, x: Tensor, context: Z, /) -> Tensor:
        r"""Alias for `forward` method."""
        return self.forward(x, context)

    def decode(self, y: Tensor, context: Z, /) -> Tensor:
        r"""Alias for `right_inverse` method."""
        return self.right_inverse(y, context)


class ConditionalBijectionBase[Z](
    nn.Module,
    ConditionalBijection[Tensor, Tensor, Z],
):
    r"""Base class for conditional bijections operating on single tensor."""

    def __invert__(self) -> ConditionalBijectionBase[Z]:
        return InverseConditionalBijection(self)

    @abstractmethod
    def forward(self, x: Tensor, context: Z, /) -> Tensor: ...
    @abstractmethod
    def inverse(self, y: Tensor, context: Z, /) -> Tensor: ...

    def encode(self, x: Tensor, context: Z, /) -> Tensor:
        return self.forward(x, context)

    def decode(self, y: Tensor, context: Z, /) -> Tensor:
        return self.inverse(y, context)


class ConditionalTransformBase[Z](
    ConditionalBijectionBase[Z],
    ConditionalTransform[Tensor, Tensor, Z],
):
    r"""Base class for transforms operating on single tensor."""

    def __invert__(self) -> ConditionalTransformBase[Z]:
        return InverseConditionalTransform(self)

    @abstractmethod
    def encode_and_logabsdet(
        self, x: Tensor, context: Z, /
    ) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def decode_and_logabsdet(
        self, y: Tensor, context: Z, /
    ) -> tuple[Tensor, Tensor]: ...

    def encode(self, x: Tensor, context: Z, /) -> Tensor:
        y, _ = self.encode_and_logabsdet(x, context)
        return y

    def decode(self, y: Tensor, context: Z, /) -> Tensor:
        x, _ = self.decode_and_logabsdet(y, context)
        return x

    def forward(self, x: Tensor, context: Z, /) -> Tensor:
        return self.encode(x, context)

    def inverse(self, y: Tensor, context: Z, /) -> Tensor:
        return self.decode(y, context)


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


class InverseConditionalBijection[
    B: ConditionalBijectionBase,
    Z,
](ConditionalBijectionBase):
    r"""Inverse of a bijection."""

    bijection: B
    r"""The bijection to be inverted."""

    def __init__(self, bijection: B) -> None:
        super().__init__()
        self.bijection = bijection

    def forward(self, x: Tensor, context: Z, /) -> Tensor:
        return self.bijection.inverse(x, context)

    def inverse(self, y: Tensor, context: Z, /) -> Tensor:
        return self.bijection.forward(y, context)


class InverseConditionalTransform[
    T: ConditionalTransformBase,
    Z,
](ConditionalTransformBase, ModuleSequence[T]):
    r"""Inverse of a transform."""

    transform: T
    r"""The transform to be inverted."""

    def __init__(self, transform: T) -> None:
        super().__init__()
        self.transform = transform

    def encode(self, x: Tensor, context: Z, /) -> Tensor:
        return self.transform.decode(x, context)

    def decode(self, y: Tensor, context: Z, /) -> Tensor:
        return self.transform.encode(y, context)

    def encode_and_logabsdet(self, x: Tensor, context: Z, /) -> tuple[Tensor, Tensor]:
        y, logabsdet = self.transform.decode_and_logabsdet(x, context)
        return y, -logabsdet

    def decode_and_logabsdet(self, y: Tensor, context: Z, /) -> tuple[Tensor, Tensor]:
        x, logabsdet = self.transform.encode_and_logabsdet(y, context)
        return x, -logabsdet


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

    def forward(self, x: Tensor, /) -> Tensor:
        for layer in self:
            x = layer.forward(x)
        return x

    def inverse(self, y: Tensor, /) -> Tensor:
        for layer in reversed(self):
            y = layer.inverse(y)
        return y


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

    def encode(self, x: Tensor, /) -> Tensor:
        for layer in self:
            x = layer.encode(x)
        return x

    def decode(self, y: Tensor, /) -> Tensor:
        for layer in reversed(self):
            y = layer.decode(y)
        return y

    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in self:
            x, logabsdet = layer.encode_and_logabsdet(x)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return x, logabsdet

    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in reversed(self):
            y, logabsdet = layer.decode_and_logabsdet(y)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return y, logabsdet


class ConditionalBijectionSequence[
    B: ConditionalBijectionBase,
    Z,
](ConditionalBijectionBase[Z], ModuleSequence[B]):
    r"""Apply multiple bijections sequentially."""

    # noinspection PyMissingConstructor
    def __init__(self, modules: Iterable[B] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[B].__init__(self, modules)

    def __invert__(self) -> ConditionalBijectionSequence[Any, Z]:
        if type(self) is not ConditionalBijectionSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )
        return ConditionalBijectionSequence(~layer for layer in reversed(self))

    def forward(self, x: Tensor, context: Z, /) -> Tensor:
        for layer in self:
            x = layer.forward(x, context)
        return x

    def inverse(self, y: Tensor, context: Z, /) -> Tensor:
        for layer in reversed(self):
            y = layer.inverse(y, context)
        return y


class ConditionalTransformSequence[
    T: ConditionalTransformBase,
    Z,
](ConditionalTransformBase, ModuleSequence[T]):
    r"""Apply multiple transforms sequentially."""

    # noinspection PyMissingConstructor
    def __init__(self, modules: Iterable[T] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[T].__init__(self, modules)

    def __invert__(self) -> ConditionalTransformSequence[Any, Z]:
        if type(self) is not ConditionalTransformSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )
        return ConditionalTransformSequence(~layer for layer in reversed(self))

    def encode(self, x: Tensor, context: Z, /) -> Tensor:
        for layer in self:
            x = layer.encode(x, context)
        return x

    def decode(self, y: Tensor, context: Z, /) -> Tensor:
        for layer in reversed(self):
            y = layer.decode(y, context)
        return y

    def encode_and_logabsdet(self, x: Tensor, context: Z, /) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in self:
            x, logabsdet = layer.encode_and_logabsdet(x, context)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return x, logabsdet

    def decode_and_logabsdet(self, y: Tensor, context: Z, /) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in reversed(self):
            y, logabsdet = layer.decode_and_logabsdet(y, context)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return y, logabsdet
