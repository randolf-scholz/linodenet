r"""Protocol types / interfaces for linodenet.mappings."""

__all__ = [
    # unconditional mappings
    "Embedding",
    "Surjection",
    "Bijection",
    "Projection",
    "Transform",
    # conditional mappings
    "ConditionalBijection",
    "ConditionalSurjection",
    "ConditionalProjection",
    "ConditionalEmbedding",
    "ConditionalTransform",
]


from abc import abstractmethod
from typing import Protocol, final, runtime_checkable

from torch import Tensor

from signatures import signature


@runtime_checkable
class Embedding[X = Tensor, Y = Tensor](Protocol):
    r"""Protocol for Embedding Components.

    See Also:
        - `ConditionalEmbedding`: Conditional embedding protocol.
        - `Surjection`: Dual protocol with a right inverse.
        - `Bijection`: Protocol combining embedding and surjection.
        - `Projection`: Idempotent self-map with identity right inverse.
        - `Transform`: Bijection with logabsdet for diffeomorphisms.
    """

    @abstractmethod
    @signature("(...) -> (...)")
    def __call__(self, x: X, /) -> Y: ...

    @abstractmethod
    @signature("(...) -> (...)")
    def left_inverse(self, y: Y, /) -> X: ...


@runtime_checkable
class Surjection[X = Tensor, Y = Tensor](Protocol):
    r"""A protocol for surjections.

    See Also:
        - `ConditionalSurjection`: Conditional surjection protocol.
        - `Embedding`: Dual protocol with a left inverse.
        - `Bijection`: Protocol combining surjection and embedding.
        - `Projection`: Idempotent self-map with identity right inverse.
        - `Transform`: Bijection with logabsdet for diffeomorphisms.
    """

    @abstractmethod
    def __call__(self, x: X, /) -> Y: ...
    @abstractmethod
    def right_inverse(self, y: Y, /) -> X: ...


@runtime_checkable
class Projection[T = Tensor](Surjection[T, T], Protocol):
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

    See Also:
        - `ConditionalProjection`: Conditional projection protocol.
        - `Surjection`: More general protocol with a right inverse.
        - `Embedding`: Dual protocol with a left inverse.
        - `Bijection`: Invertible protocol combining both inverses.
        - `Transform`: Diffeomorphism protocol, stronger than a projection.
    """

    @abstractmethod
    @signature("(..., *xs) -> (..., *ys)")
    def __call__(self, x: T, /) -> T: ...

    @final
    @signature("(..., *ys) -> (..., *xs)")
    def right_inverse(self, y: T, /) -> T:
        r"""Right inverse of the projection, i.e. the identity on the image."""
        return y


@runtime_checkable
class Bijection[X = Tensor, Y = Tensor](Surjection[X, Y], Embedding[X, Y], Protocol):
    r"""Protocol for invertible layers.

    See Also:
        - `ConditionalBijection`: Conditional bijection protocol.
        - `Embedding`: Protocol providing a left inverse.
        - `Surjection`: Protocol providing a right inverse.
        - `Projection`: Idempotent self-map with identity right inverse.
        - `Transform`: Bijection with logabsdet for diffeomorphisms.
    """

    @abstractmethod
    def __call__(self, x: X, /) -> Y: ...
    @abstractmethod
    def inverse(self, y: Y, /) -> X: ...

    def right_inverse(self, y: Y, /) -> X:
        return self.inverse(y)

    def left_inverse(self, y: Y, /) -> X:
        return self.inverse(y)


@runtime_checkable
class Transform[X = Tensor, Y = Tensor](Bijection[X, Y], Protocol):
    r"""Protocol for diffeomorphism with logabsdet.

    Note:
        A diffeomorphism is a bijection $f:X→Y$ such that both $f$ and $f⁻¹$
        are differentiable.

        A differentiable bijection need not be a diffeomorphism. A standard
        example is $f:ℝ→ℝ$, $f(x)=x³$, which is a differentiable bijection, but
        its inverse $f⁻¹(y)=∛y$ is not differentiable at $y=0$.

    See Also:
        - `ConditionalTransform`: Conditional diffeomorphism protocol.
        - `Bijection`: Underlying invertible protocol without logabsdet.
        - `Embedding`: Protocol providing a left inverse.
        - `Surjection`: Protocol providing a right inverse.
        - `Projection`: Idempotent self-map protocol.
    """

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

    def __call__(self, x: X, /) -> Y:
        return self.encode(x)

    def inverse(self, y: Y, /) -> X:
        return self.decode(y)


@runtime_checkable
class ConditionalEmbedding[X = Tensor, Y = Tensor, Z = Tensor](Protocol):
    r"""Protocol for Conditional Embedding Components.

    See Also:
        - `Embedding`: Unconditional embedding protocol.
        - `ConditionalSurjection`: Conditional protocol with a right inverse.
        - `ConditionalBijection`: Conditional protocol combining both inverses.
        - `ConditionalProjection`: Conditional idempotent self-map protocol.
        - `ConditionalTransform`: Conditional diffeomorphism protocol.
    """

    @abstractmethod
    def condition(self, context: Z, /) -> Embedding[X, Y]:
        r"""Condition on a fixed context to get an unconditional Embedding."""
        ...

    @abstractmethod
    @signature("(...) -> (...)")
    def __call__(self, x: X, context: Z, /) -> Y: ...

    @abstractmethod
    @signature("(...) -> (...)")
    def left_inverse(self, y: Y, context: Z, /) -> X: ...


@runtime_checkable
class ConditionalSurjection[X = Tensor, Y = Tensor, Z = Tensor](Protocol):
    r"""A protocol for conditional surjections.

    See Also:
        - `Surjection`: Unconditional surjection protocol.
        - `ConditionalEmbedding`: Conditional protocol with a left inverse.
        - `ConditionalBijection`: Conditional protocol combining both inverses.
        - `ConditionalProjection`: Conditional idempotent self-map protocol.
        - `ConditionalTransform`: Conditional diffeomorphism protocol.
    """

    @abstractmethod
    def condition(self, context: Z, /) -> Surjection[X, Y]:
        r"""Condition on a fixed context to get an unconditional surjection."""
        ...

    @abstractmethod
    @signature("(...) -> (...)")
    def __call__(self, x: X, context: Z, /) -> Y: ...

    @abstractmethod
    @signature("(...) -> (...)")
    def right_inverse(self, y: Y, context: Z, /) -> X: ...


@runtime_checkable
class ConditionalProjection[T = Tensor, Z = Tensor](
    ConditionalSurjection[T, T, Z], Protocol
):
    r"""Protocol for conditional projections.

    For every fixed context, a conditional projection is idempotent in its input.
    That is, $φ(φ(x, z), z)=φ(x, z)$ and the right inverse is the identity on the
    image for that context.

    See Also:
        - `Projection`: Unconditional projection protocol.
        - `ConditionalSurjection`: More general conditional protocol with a right inverse.
        - `ConditionalEmbedding`: Dual conditional protocol with a left inverse.
        - `ConditionalBijection`: Conditional invertible protocol combining both inverses.
        - `ConditionalTransform`: Conditional diffeomorphism protocol, stronger than a projection.
    """

    @abstractmethod
    def condition(self, context: Z, /) -> Projection[T]:
        r"""Condition on a fixed context to get an unconditional projection."""
        ...

    @abstractmethod
    @signature("(..., *xs) -> (..., *ys)")
    def __call__(self, x: T, context: Z, /) -> T: ...

    @signature("(..., *ys) -> (..., *xs)")
    def right_inverse(self, y: T, context: Z, /) -> T:  # noqa: ARG002
        r"""Right inverse of the conditional projection for a fixed context."""
        return y


@runtime_checkable
class ConditionalBijection[X = Tensor, Y = Tensor, Z = Tensor](
    ConditionalSurjection[X, Y, Z],
    ConditionalEmbedding[X, Y, Z],
    Protocol,
):
    r"""Protocol for conditional invertible layers.

    See Also:
        - `Bijection`: Unconditional bijection protocol.
        - `ConditionalEmbedding`: Conditional protocol providing a left inverse.
        - `ConditionalSurjection`: Conditional protocol providing a right inverse.
        - `ConditionalProjection`: Conditional idempotent self-map protocol.
        - `ConditionalTransform`: Conditional bijection with logabsdet.
    """

    @abstractmethod
    def condition(self, context: Z, /) -> Bijection[X, Y]:
        r"""Condition on a fixed context to get an unconditional bijection."""
        ...

    @abstractmethod
    @signature("(...) -> (...)")
    def __call__(self, x: X, context: Z, /) -> Y: ...

    @abstractmethod
    @signature("(...) -> (...)")
    def inverse(self, y: Y, context: Z, /) -> X: ...

    def right_inverse(self, y: Y, context: Z, /) -> X:
        return self.inverse(y, context)

    def left_inverse(self, y: Y, context: Z, /) -> X:
        return self.inverse(y, context)


@runtime_checkable
class ConditionalTransform[X = Tensor, Y = Tensor, Z = Tensor](
    ConditionalBijection[X, Y, Z], Protocol
):
    r"""Protocol for diffeomorphism with logabsdet.

    See Also:
        - `Transform`: Unconditional diffeomorphism protocol.
        - `ConditionalBijection`: Underlying conditional invertible protocol.
        - `ConditionalEmbedding`: Conditional protocol providing a left inverse.
        - `ConditionalSurjection`: Conditional protocol providing a right inverse.
        - `ConditionalProjection`: Conditional idempotent self-map protocol.
    """

    @abstractmethod
    def condition(self, context: Z, /) -> Transform[X, Y]:
        r"""Condition on a fixed context to get an unconditional transform."""
        ...

    @abstractmethod
    def encode_and_logabsdet(self, x: X, context: Z, /) -> tuple[Y, Tensor]: ...
    @abstractmethod
    def decode_and_logabsdet(self, y: Y, context: Z, /) -> tuple[X, Tensor]: ...

    def encode(self, x: X, context: Z, /) -> Y:
        y, _ = self.encode_and_logabsdet(x, context)
        return y

    def decode(self, y: Y, context: Z, /) -> X:
        x, _ = self.decode_and_logabsdet(y, context)
        return x
