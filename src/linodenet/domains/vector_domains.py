r"""Vector-specific domain labels and their partial-order relations."""

__all__ = ["VectorDomains"]


from types import MappingProxyType

from torch import Tensor

from .base import PosetEnum


class VectorDomains(PosetEnum):
    r"""Enumeration of some vector domains."""

    ANY = "any"  # top node
    NONE = "none"  # bottom node

    REAL = "real"
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
    UNIT_CUBE = "unit-cube"  # ‖x‖_∞ ≤ 1
    UNIT_L1BALL = "unit-l1ball"  # ‖x‖₁ ≤ 1
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
    V.NEGATIVE: {V.REAL & V.NONPOSITIVE & V.NONZERO},
    V.NONNEGATIVE: {V.REAL},
    V.NONPOSITIVE: {V.REAL},
    V.ONE: {V.BOOLEAN},
    V.ONE_HOT: {V.SPARSE},
    V.POSITIVE: {V.REAL & V.NONNEGATIVE & V.NONZERO},
    V.REAL: {V.COMPLEX},
    V.STANDARDIZED: {V.ZERO_MEAN & V.NONZERO},
    V.STOCHASTIC: {V.NONNEGATIVE & V.NONZERO & V.UNIT_L1BALL},
    V.UNIT_VECTOR: {V.NONZERO & V.UNIT_BALL},
    V.UNIT_BALL: {V.UNIT_CUBE},
    V.UNIT_L1BALL: {V.UNIT_BALL},
    V.ZERO: {V.SPARSE, V.BOOLEAN},
})  # fmt: skip
VectorDomains.KNOWN_SUBTYPES = MappingProxyType({})
del V  # remove alias
