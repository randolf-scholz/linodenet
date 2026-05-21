r"""Base classes for mappings."""

__all__ = [
    # Classes
    "BijectionBase",
    "EmbeddingBase",
    "ProjectionBase",
    "SurjectionBase",
    "TransformBase",
    # sequence classes
    "BijectionSequence",
    "TransformSequence",
    # sequence + conditional
    "ConditionalBijectionSequence",
    "ConditionalTransformSequence",
]


from abc import abstractmethod
from collections.abc import Iterable, Iterator
from typing import final

import torch
from torch import Tensor, nn

from linodenet.nn import ModuleSequence
from signatures import signature

from .abstract import (
    Bijection,
    ConditionalBijection,
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


class SurjectionBase(nn.Module, Surjection[Tensor, Tensor]):
    r"""Abstract Base Class for Surjection components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def right_inverse(self, y: Tensor, /) -> Tensor: ...


class ProjectionBase(SurjectionBase, Projection[Tensor]):
    r"""Abstract Base Class for Projection components."""

    @abstractmethod
    @signature("(..., *xs) -> (..., *ys)")
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the projection."""

    @final
    @signature("(..., *ys) -> (..., *xs)")
    def right_inverse(self, y: Tensor, /) -> Tensor:
        r"""Right inverse of the projection, i.e. the identity on the image."""
        return y


class BijectionBase(nn.Module, Bijection[Tensor, Tensor]):
    r"""Base class for bijections operating on single tensor."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor: ...
    @abstractmethod
    def inverse(self, y: Tensor, /) -> Tensor: ...


class TransformBase(BijectionBase, Transform[Tensor, Tensor]):
    r"""Base class for transforms operating on single tensor."""

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


# TODO: use intersection type for upper bound `Bijection & nn.Module`
class BijectionSequence[
    B: Bijection,
](ModuleSequence[B], Bijection):  # type: ignore[bad-specialization]
    r"""Apply multiple bijections sequentially."""

    # noinspection PyMissingConstructor
    def __init__(self, modules: Iterable[B] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[B].__init__(self, modules)  # type: ignore[bad-specialization]

    def __invert__(self) -> BijectionSequence:
        if type(self) is not BijectionSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )

        def _inverse_iterator() -> Iterator[TransformBase]:
            for i, layer in enumerate(reversed(self)):
                try:
                    yield ~layer
                except (TypeError, NotImplementedError) as exc:
                    raise NotImplementedError(
                        f"Inversion not implemented for layer {i} of type {type(layer)}"
                    ) from exc

        return BijectionSequence(_inverse_iterator())

    def forward(self, x: Tensor, /) -> Tensor:
        for layer in self:
            x = layer(x)
        return x

    def inverse(self, y: Tensor, /) -> Tensor:
        for layer in reversed(self):
            y = layer.inverse(y)
        return y


# TODO: use intersection type for upper bound `Transform & nn.Module`
class TransformSequence[
    T: Transform,
](ModuleSequence[T], Transform):  # type: ignore[bad-specialization]
    r"""Apply multiple transforms sequentially."""

    # noinspection PyMissingConstructor
    def __init__(self, modules: Iterable[T] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[T].__init__(self, modules)  # type: ignore[bad-specialization]

    def __invert__(self) -> TransformSequence:
        if type(self) is not TransformSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )

        def _inverse_iterator() -> Iterator[TransformBase]:
            for i, layer in enumerate(reversed(self)):
                try:
                    yield ~layer
                except (TypeError, NotImplementedError) as exc:
                    raise NotImplementedError(
                        f"Inversion not implemented for layer {i} of type {type(layer)}"
                    ) from exc

        return TransformSequence(_inverse_iterator())

    def forward(self, x: Tensor, /) -> Tensor:
        return self.encode(x)

    def inverse(self, y: Tensor, /) -> Tensor:
        return self.decode(y)

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


# TODO: use intersection type for upper bound `ConditionalBijection & nn.Module`
class ConditionalBijectionSequence[
    B: ConditionalBijection,
](ModuleSequence[B], ConditionalBijection):  # type: ignore[bad-specialization]
    r"""Apply multiple bijections sequentially."""

    # noinspection PyMissingConstructor
    def __init__(self, modules: Iterable[B] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[B].__init__(self, modules)  # type: ignore[bad-specialization]

    def __invert__(self) -> ConditionalBijectionSequence:
        if type(self) is not ConditionalBijectionSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )

        def _inverse_iterator() -> Iterator[ConditionalBijection]:
            for i, layer in enumerate(reversed(self)):
                try:
                    yield ~layer
                except (TypeError, NotImplementedError) as exc:
                    raise NotImplementedError(
                        f"Inversion not implemented for layer {i} of type {type(layer)}"
                    ) from exc

        return ConditionalBijectionSequence(_inverse_iterator())

    def condition(self, context: Tensor, /) -> BijectionSequence:
        return BijectionSequence(layer.condition(context) for layer in self)

    def forward(self, x: Tensor, context: Tensor, /) -> Tensor:
        for layer in self:
            x = layer(x, context)
        return x

    def inverse(self, y: Tensor, context: Tensor, /) -> Tensor:
        for layer in reversed(self):
            y = layer.inverse(y, context)
        return y


# TODO: use intersection type for upper bound `ConditionalTransform & nn.Module`
class ConditionalTransformSequence[
    T: ConditionalTransform,
](ModuleSequence[T], ConditionalTransform):  # type: ignore[bad-specialization]
    r"""Apply multiple transforms sequentially."""

    # noinspection PyMissingConstructor
    def __init__(self, modules: Iterable[T] = (), /) -> None:
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[T].__init__(self, modules)  # type: ignore[bad-specialization]

    def __invert__(self) -> ConditionalTransformSequence:
        if type(self) is not ConditionalTransformSequence:
            raise NotImplementedError(
                f"Inversion not implemented for subclass {type(self)}"
            )

        def _inverse_iterator() -> Iterator[ConditionalTransform]:
            for i, layer in enumerate(reversed(self)):
                try:
                    yield ~layer
                except (TypeError, NotImplementedError) as exc:
                    raise NotImplementedError(
                        f"Inversion not implemented for layer {i} of type {type(layer)}"
                    ) from exc

        return ConditionalTransformSequence(_inverse_iterator())

    def condition(self, context: Tensor, /) -> TransformSequence:
        return TransformSequence(layer.condition(context) for layer in self)

    def inverse(self, y: Tensor, context: Tensor, /) -> Tensor:
        return self.decode(y, context)

    def encode(self, x: Tensor, context: Tensor, /) -> Tensor:
        for layer in self:
            x = layer.encode(x, context)
        return x

    def decode(self, y: Tensor, context: Tensor, /) -> Tensor:
        for layer in reversed(self):
            y = layer.decode(y, context)
        return y

    def encode_and_logabsdet(
        self, x: Tensor, context: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in self:
            x, logabsdet = layer.encode_and_logabsdet(x, context)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return x, logabsdet

    def decode_and_logabsdet(
        self, y: Tensor, context: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        logabsdets: list[Tensor] = []

        for layer in reversed(self):
            y, logabsdet = layer.decode_and_logabsdet(y, context)
            logabsdets.append(logabsdet)

        logabsdet = torch.stack(logabsdets, dim=-1).sum(dim=-1)
        return y, logabsdet
