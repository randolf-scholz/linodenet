r"""Vector-specific domain primitives and partial-order labels."""

__all__ = [
    "VectorDomains",
    "Boolean",
    "Complex",
    "Discrete",
    "Empty",
    "One",
    "Sparse",
    "Vector",
    "OneHot",
    "NonZero",
    "Positive",
    "Real",
    "Negative",
    "NonNegative",
    "NonPositive",
    "Stochastic",
    "Standardized",
    "UnitBall",
    "UnitCube",
    "UnitL1Ball",
    "UnitL1Sphere",
    "UnitVector",
    "Zero",
    "ZeroMean",
]

from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import Final, Self, overload

import torch
from torch import Tensor

from . import vector_tests as tests
from .base import PosetEnum, VectorDomain


@dataclass(frozen=True)
class Vector(VectorDomain):
    r"""Domain of vectors with optional fixed size."""

    size: Final[int | None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def check(self, value: Tensor, /) -> Tensor:
        *batch_shape, n = value.shape
        if self.size is None:
            return value.new_full(batch_shape, True, dtype=torch.bool)
        return value.new_full(batch_shape, n == self.size, dtype=torch.bool)

    @overload
    def __call__(self, /) -> Self: ...
    @overload
    def __call__(self, size: int, /) -> Self: ...
    def __call__(self, size: int | None = None, /) -> Self:
        return self.__class__(size)


@dataclass(frozen=True)
class Complex(Vector):
    r"""Domain of vectors with values in the complex numbers."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value)


@dataclass(frozen=True)
class Real(Complex):
    r"""Domain of vectors with real dtype."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_real_vector(value)


@dataclass(frozen=True)
class Discrete(Real):
    r"""Domain of vectors with integer or boolean dtype."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_discrete_vector(value)


@dataclass(frozen=True)
class Boolean(Vector):
    r"""Domain of vectors whose entries are only zeros and ones."""

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & tests.is_boolean_vector(value)


@dataclass(frozen=True)
class Empty(Vector):
    r"""Domain of vectors with no admissible values."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.new_full(value.shape[:-1], False, dtype=torch.bool)


@dataclass(frozen=True)
class Zero(Vector):
    r"""Domain of vectors whose entries are only zeros."""

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & tests.is_zero_vector(value)


@dataclass(frozen=True)
class One(Vector):
    r"""Domain of vectors whose entries are only ones."""

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & tests.is_one_vector(value)


@dataclass(frozen=True)
class Sparse(Vector):
    r"""Domain of vectors with sufficiently many exact zero entries."""

    _: KW_ONLY
    sparsity: Final[float | None] = None

    def __post_init__(self) -> None:
        if self.sparsity is not None and not 0.0 <= self.sparsity <= 1.0:
            raise ValueError("Expected sparsity in [0, 1].")

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_sparse_vector(value, self.sparsity)

    @overload
    def __call__(self, /) -> Self: ...
    @overload
    def __call__(self, size: int, /, *, sparsity: float | None = None) -> Self: ...
    def __call__(
        self, size: int | None = None, /, *, sparsity: float | None = None
    ) -> Self:
        return self.__class__(size, sparsity=sparsity)


@dataclass(frozen=True)
class NonZero(Vector):
    r"""Domain of vectors that are not identically zero."""

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & tests.is_nonzero_vector(value)


@dataclass(frozen=True)
class NonNegative(Vector):
    r"""Domain of entrywise nonnegative vectors."""

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & tests.is_nonnegative_vector(value)


@dataclass(frozen=True)
class NonPositive(Vector):
    r"""Domain of entrywise nonpositive vectors."""

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & tests.is_nonpositive_vector(value)


@dataclass(frozen=True)
class Positive(NonNegative):
    r"""Domain of strictly positive vectors."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_positive_vector(value)


@dataclass(frozen=True)
class Negative(NonPositive):
    r"""Domain of strictly negative vectors."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_negative_vector(value)


@dataclass(frozen=True)
class Stochastic(NonNegative):
    r"""Domain of nonnegative vectors whose entries sum to one."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_stochastic_vector(value)


@dataclass(frozen=True)
class OneHot(Stochastic):
    r"""Domain of one-hot vectors."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_one_hot_vector(value)


@dataclass(frozen=True)
class ZeroMean(Vector):
    r"""Domain of vectors with zero empirical mean."""

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & tests.is_zero_mean_vector(value)


@dataclass(frozen=True)
class Standardized(ZeroMean):
    r"""Domain of vectors with zero mean and unit variance."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_standardized_vector(value)


@dataclass(frozen=True)
class UnitVector(NonZero):
    r"""Domain of vectors with Euclidean norm equal to one."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_unit_vector(value)


@dataclass(frozen=True)
class UnitCube(Vector):
    r"""Domain of vectors in the ℓ∞ unit ball."""

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & tests.is_unit_cube_vector(value)


@dataclass(frozen=True)
class UnitBall(UnitCube):
    r"""Domain of vectors in the Euclidean unit ball."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_unit_ball_vector(value)


@dataclass(frozen=True)
class UnitL1Ball(UnitBall):
    r"""Domain of vectors in the ℓ¹ unit ball."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_unit_l1_ball_vector(value)


@dataclass(frozen=True)
class UnitL1Sphere(UnitL1Ball):
    r"""Domain of vectors with ℓ¹ norm equal to one."""

    def check(self, value: Tensor, /) -> Tensor:
        return Vector.check(self, value) & tests.is_unit_l1_sphere_vector(value)


class VectorDomains(PosetEnum):
    r"""Enumeration of some vector domains."""

    ANY = "any"  # top node
    NONE = "none"  # bottom node

    REAL = "real"
    DISCRETE = "discrete"
    COMPLEX = "complex"
    BOOLEAN = "boolean"

    # specific vectors
    ZERO = "zero"  # xᵢ=0 for all i
    ONE = "one"  # xᵢ=1 for all i

    SPARSE = "sparse"  # xᵢ=0 for many i
    ONE_HOT = "one-hot"  # xᵢ=1, xⱼ=0 for j≠i

    ZERO_MEAN = "zero-mean"
    STANDARDIZED = "standardized"  # zero-mean, unit variance

    UNIT_VECTOR = "unit-vector"  # ‖x‖₂ = 1
    UNIT_SPHERE = "unit-vector"  # alias
    UNIT_BALL = "unit-ball"  # ‖x‖₂ ≤ 1

    UNIT_L1_BALL = "unit-l1-ball"  # ‖x‖₁ ≤ 1
    UNIT_L1_SPHERE = "unit-l1-sphere"  # ‖x‖₁ = 1

    UNIT_CUBE = "unit-cube"  # ‖x‖_∞ ≤ 1

    STOCHASTIC = "stochastic"  # ∑xᵢ = 1, xᵢ ≥ 0
    SIMPLEX = "stochastic"  # alias

    NONZERO = "nonzero"  # x ≠ 0
    POSITIVE = "positive"  # xᵢ > 0
    NEGATIVE = "negative"  # xᵢ < 0
    NONNEGATIVE = "nonnegative"  # xᵢ ≥ 0
    NONPOSITIVE = "nonpositive"  # xᵢ ≤ 0

    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError


V = VectorDomains  # temporary alias
VectorDomains.KNOWN_MEETS = (
    (V.NONE, V.NEGATIVE & V.NONNEGATIVE),
    (V.NONE, V.NONZERO & V.ZERO),
    (V.NONE, V.POSITIVE & V.NONPOSITIVE),
    (V.NONE, V.POSITIVE & V.NEGATIVE),
    (V.ONE_HOT, V.BOOLEAN & V.STOCHASTIC),
    (V.ONE_HOT, V.STOCHASTIC & V.UNIT_VECTOR),
    (V.ZERO, V.NONNEGATIVE & V.NONPOSITIVE),
)
VectorDomains.KNOWN_SUPERTYPES = MappingProxyType({
    V.BOOLEAN: {V.REAL & V.NONNEGATIVE},
    V.DISCRETE: {V.REAL},
    V.NEGATIVE: {V.REAL & V.NONPOSITIVE & V.NONZERO},
    V.NONNEGATIVE: {V.REAL},
    V.NONPOSITIVE: {V.REAL},
    V.ONE: {V.NONZERO},
    V.ONE_HOT: {V.SPARSE},
    V.POSITIVE: {V.REAL & V.NONNEGATIVE & V.NONZERO},
    V.REAL: {V.COMPLEX},
    V.STANDARDIZED: {V.ZERO_MEAN & V.NONZERO},
    V.STOCHASTIC: {V.NONNEGATIVE & V.NONZERO & V.UNIT_L1_BALL},
    V.UNIT_VECTOR: {V.NONZERO & V.UNIT_BALL},
    V.UNIT_BALL: {V.UNIT_CUBE},
    V.UNIT_L1_BALL: {V.UNIT_BALL},
    V.ZERO: {V.BOOLEAN},
})  # fmt: skip
VectorDomains.KNOWN_SUBTYPES = MappingProxyType(
    {V.SPARSE: {V.ZERO, V.ONE_HOT}},
)
del V  # remove alias
