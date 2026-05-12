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

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final, Self, overload

import torch
from torch import Tensor

from . import VectorDomain, vector_tests as tests
from .base import Indeterminate, PosetEnum


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


class VectorDomains(VectorDomain, PosetEnum):
    r"""Enumeration of some vector domains."""

    ALIASES: ClassVar[Mapping[str, Self]]

    ANY = Vector()  # top node
    NONE = Empty()  # bottom node

    REAL = Real()
    DISCRETE = Discrete()
    COMPLEX = Complex()
    BOOLEAN = Boolean()

    # specific vectors
    ZERO = Zero()  # xᵢ=0 for all i
    ONE = One()  # xᵢ=1 for all i

    SPARSE = Sparse()  # xᵢ=0 for many i
    ONE_HOT = OneHot()  # xᵢ=1, xⱼ=0 for j≠i

    ZERO_MEAN = ZeroMean()
    STANDARDIZED = Standardized()  # zero-mean, unit variance

    UNIT_VECTOR = UnitVector()  # ‖x‖₂ = 1
    UNIT_SPHERE = UnitVector()  # alias
    UNIT_BALL = UnitBall()  # ‖x‖₂ ≤ 1

    UNIT_L1_BALL = UnitL1Ball()  # ‖x‖₁ ≤ 1
    UNIT_L1_SPHERE = UnitL1Sphere()  # ‖x‖₁ = 1

    UNIT_CUBE = UnitCube()  # ‖x‖_∞ ≤ 1

    STOCHASTIC = Stochastic()  # ∑xᵢ = 1, xᵢ ≥ 0
    SIMPLEX = Stochastic()  # alias

    NONZERO = NonZero()  # x ≠ 0
    POSITIVE = Positive()  # xᵢ > 0
    NEGATIVE = Negative()  # xᵢ < 0
    NONNEGATIVE = NonNegative()  # xᵢ ≥ 0
    NONPOSITIVE = NonPositive()  # xᵢ ≤ 0

    @property
    def size(self) -> int | None:
        return self.value.size

    def check(self, value: Tensor, /) -> Tensor:
        return self.value.check(value)

    def __le__(self, other: Any, /) -> bool | Indeterminate:
        return PosetEnum.__le__(self, other)

    def __lt__(self, other: Any, /) -> bool | Indeterminate:
        return PosetEnum.__lt__(self, other)

    def __ge__(self, other: Any, /) -> bool | Indeterminate:
        return PosetEnum.__ge__(self, other)

    def __gt__(self, other: Any, /) -> bool | Indeterminate:
        return PosetEnum.__gt__(self, other)

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, str):
            return cls.ALIASES.get(value)
        return None


V = VectorDomains  # temporary alias
VectorDomains.KNOWN_MEETS = (  # type: ignore[assignment]  # pyrefly: ignore[bad-assignment]
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
VectorDomains.ALIASES = MappingProxyType({
    "any"            : V.ANY,
    "boolean"        : V.BOOLEAN,
    "complex"        : V.COMPLEX,
    "discrete"       : V.DISCRETE,
    "empty"          : V.NONE,
    "negative"       : V.NEGATIVE,
    "none"           : V.NONE,
    "nonnegative"    : V.NONNEGATIVE,
    "nonpositive"    : V.NONPOSITIVE,
    "nonzero"        : V.NONZERO,
    "one"            : V.ONE,
    "one-hot"        : V.ONE_HOT,
    "positive"       : V.POSITIVE,
    "real"           : V.REAL,
    "simplex"        : V.SIMPLEX,
    "sparse"         : V.SPARSE,
    "standardized"   : V.STANDARDIZED,
    "stochastic"     : V.STOCHASTIC,
    "unit-ball"      : V.UNIT_BALL,
    "unit-cube"      : V.UNIT_CUBE,
    "unit-l1-ball"   : V.UNIT_L1_BALL,
    "unit-l1-sphere" : V.UNIT_L1_SPHERE,
    "unit-sphere"    : V.UNIT_SPHERE,
    "unit-vector"    : V.UNIT_VECTOR,
    "zero"           : V.ZERO,
    "zero-mean"      : V.ZERO_MEAN,
})  # fmt: skip
del V  # remove alias
