r"""Type-checking coherence tests for mapping protocols."""

from abc import abstractmethod
from typing import Protocol

from torch import Tensor

from linodenet.mappings.abstract import (
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


class _Embedding[X, Y, **P](Protocol):
    @abstractmethod
    def __call__(self, x: X, /, *args: P.args, **kwargs: P.kwargs) -> Y: ...

    @abstractmethod
    def left_inverse(self, y: Y, /, *args: P.args, **kwargs: P.kwargs) -> X: ...


class _Surjection[X, Y, **P](Protocol):
    @abstractmethod
    def __call__(self, x: X, /, *args: P.args, **kwargs: P.kwargs) -> Y: ...

    @abstractmethod
    def right_inverse(self, y: Y, /, *args: P.args, **kwargs: P.kwargs) -> X: ...


class _Projection[T, **P](_Surjection[T, T, P], Protocol): ...


class _Bijection[X, Y, **P](_Surjection[X, Y, P], _Embedding[X, Y, P], Protocol):
    @abstractmethod
    def __call__(self, x: X, /, *args: P.args, **kwargs: P.kwargs) -> Y: ...

    @abstractmethod
    def inverse(self, y: Y, /, *args: P.args, **kwargs: P.kwargs) -> X: ...


class _Transform[X, Y, **P](_Bijection[X, Y, P], Protocol):
    @abstractmethod
    def encode_and_logabsdet(
        self, x: X, /, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[Y, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(
        self, y: Y, /, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[X, Tensor]: ...
    def encode(self, x: X, /, *args: P.args, **kwargs: P.kwargs) -> Y: ...
    def decode(self, y: Y, /, *args: P.args, **kwargs: P.kwargs) -> X: ...


def test_protocol_coherence[X, Y, Z](
    embedding: Embedding[X, Y],
    surjection: Surjection[X, Y],
    projection: Projection[X],
    bijection: Bijection[X, Y],
    transform: Transform[X, Y],
    cond_embedding: ConditionalEmbedding[X, Y, Z],
    cond_surjection: ConditionalSurjection[X, Y, Z],
    cond_bijection: ConditionalBijection[X, Y, Z],
    cond_projection: ConditionalProjection[X, Z],
    cond_transform: ConditionalTransform[X, Y, Z],
) -> None:
    _0: _Embedding[X, Y, []] = embedding
    _1: _Surjection[X, Y, []] = surjection
    _2: _Bijection[X, Y, []] = bijection
    _3: _Projection[X, []] = projection
    _4: _Transform[X, Y, []] = transform

    _5: _Embedding[X, Y, [Z]] = cond_embedding
    _6: _Surjection[X, Y, [Z]] = cond_surjection
    _7: _Bijection[X, Y, [Z]] = cond_bijection
    _8: _Projection[X, [Z]] = cond_projection
    _9: _Transform[X, Y, [Z]] = cond_transform
