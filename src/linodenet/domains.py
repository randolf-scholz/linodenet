r"""WORK IN PROGRESS.

Domains should allow:

1. checking membership of tensors
2. checking subset relations between domains
3. performing some basic operations (e.g. product of domains, union, intersection)
"""

__all__ = [
    "Domain",
    "Interval",
    "ScalarDomains",
    "VectorDomains",
    "MatrixDomains",
]

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import cache
from types import MappingProxyType
from typing import ClassVar, Final, Protocol, Self

from torch import Tensor


class Domain(Protocol):
    r"""Protocol for Domains."""

    def __contains__(self, item: Tensor, /) -> Tensor: ...

    def __le__(self, other: Domain, /) -> bool: ...


class _PosetEnum(Enum):
    r"""Mixin implementing a partial order from immediate-superset edges."""

    KNOWN_EDGES: ClassVar[Mapping[Self, frozenset[Self]]]

    @classmethod
    @cache
    def _validated_edges(cls) -> Mapping[Self, frozenset[Self]]:
        edges: Mapping[Self, frozenset[Self]] = cls.KNOWN_EDGES  # type: ignore[assignment]
        members = frozenset(cls)

        if bad_keys := {node for node in edges if node not in members}:
            raise TypeError(f"Expected {cls.__name__} nodes, got {bad_keys!r}.")

        all_targets = frozenset().union(*edges.values())
        if bad_targets := {target for target in all_targets if target not in members}:
            raise TypeError(f"Expected {cls.__name__} targets, got {bad_targets!r}.")

        stack: set[Self] = set()
        visited: set[Self] = set()

        def visit(node: Self, /) -> None:
            if node in stack:
                raise ValueError(f"Cycle detected in {cls.__name__} order at {node!r}.")
            if node in visited:
                return

            stack.add(node)
            for target in edges.get(node, frozenset()):
                visit(target)
            stack.remove(node)
            visited.add(node)

        for node in cls:
            visit(node)

        return edges

    @classmethod
    @cache
    def _upward_closure(cls, node: Self, /) -> frozenset[Self]:
        edges = cls._validated_edges()
        parents = edges.get(node, frozenset())

        closure = frozenset({node, *parents})
        for parent in parents:
            closure = closure | cls._upward_closure(parent)

        return closure

    def __le__(self, other: object, /) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return other in type(self)._upward_closure(self)

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class Interval(Domain):
    r"""A named tuple representing an interval."""

    lower: Final[float]
    upper: Final[float]
    lower_inclusive: Final[bool]
    upper_inclusive: Final[bool]

    def __init__(
        self,
        lower: float,
        upper: float,
        *,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.lower_inclusive = lower_inclusive
        self.upper_inclusive = upper_inclusive

    @classmethod
    def from_string(cls, s: str, /) -> Interval:
        r"""Create an Interval from a string representation.

        Examples:
            >>> Interval.from_string("[0, 1)")
            Interval(lower=0.0, upper=1.0, lower_inclusive=True, upper_inclusive=False)
            >>> Interval.from_string("(-inf, inf)")
            Interval(lower=-inf, upper=inf, lower_inclusive=False, upper_inclusive=False)
        """
        s = s.strip()

        match s[0]:
            case "[":
                lower_inclusive = True
            case "(":
                lower_inclusive = False
            case _:
                raise ValueError(f"Invalid interval string: {s}")

        match s[-1]:
            case "]":
                upper_inclusive = True
            case ")":
                upper_inclusive = False
            case _:
                raise ValueError(f"Invalid interval string: {s}")

        bounds = s[1:-1].split(",")
        if len(bounds) != 2:
            raise ValueError(f"Invalid interval string: {s}")

        lower_str, upper_str = bounds
        lower = float(lower_str.strip())
        upper = float(upper_str.strip())

        return cls(
            lower,
            upper,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        )

    def __contains__(self, item: Tensor, /) -> Tensor:
        lower_mask = (
            (item >= self.lower) if self.lower_inclusive else (item > self.lower)
        )
        upper_mask = (
            (item <= self.upper) if self.upper_inclusive else (item < self.upper)
        )
        return lower_mask & upper_mask

    def __str__(self) -> str:
        lower_bracket = "[" if self.lower_inclusive else "("
        upper_bracket = "]" if self.upper_inclusive else ")"
        return f"{lower_bracket}{self.lower}, {self.upper}{upper_bracket}"

    def __repr__(self) -> str:
        return f"Interval('{self!s}')"


class ScalarDomains:
    r"""Some scalar domains."""

    REAL_LINE: Final[Interval] = Interval.from_string("(-inf, inf)")
    EXTENDED_LINE: Final[Interval] = Interval.from_string("[-inf, inf]")
    UNIT_INTERVAL: Final[Interval] = Interval.from_string("[0, 1]")
    OPEN_UNIT_INTERVAL: Final[Interval] = Interval.from_string("(0, 1)")
    POSITIVE_REALS: Final[Interval] = Interval.from_string("(0, inf)")
    NONNEGATIVE_REALS: Final[Interval] = Interval.from_string("[0, inf)")
    NEGATIVE_REALS: Final[Interval] = Interval.from_string("(-inf, 0)")
    NONPOSITIVE_REALS: Final[Interval] = Interval.from_string("(-inf, 0]")


class VectorDomains(StrEnum):
    r"""Enumeration of some vector domains."""

    REAL = "real"
    COMPLEX = "complex"
    BOOLEAN = "boolean"

    SPARSE = "sparse"  # xᵢ=0 for many i
    ONE_HOT = "one-hot"  # xᵢ=1, xⱼ=0 for j≠i

    ZERO_MEAN = "zero-mean"
    STANDARDIZED = "standardized"  # zero-mean, unit variance
    NORMALIZED = "normalized"  # min=0, max=1

    UNIT_VECTOR = "unit_sphere"  # ‖x‖₂ = 1
    STOCHASTIC = "stochastic"  # sum(x) = 1, x ≥ 0

    NONZERO = "nonzero"  # x ≠ 0
    POSITIVE = "positive"  # xᵢ > 0
    NEGATIVE = "negative"  # xᵢ < 0
    NONNEGATIVE = "nonnegative"  # xᵢ ≥ 0
    NONPOSITIVE = "nonpositive"  # xᵢ ≤ 0


class MatrixDomains(_PosetEnum):
    r"""Enumeration of some matrix domains."""

    RECTANGULAR = "rectangular"  # m × n matrices
    SQUARE = "square"  # n × n matrices
    EVEN_SQUARE = "even_square"  # 2n × 2n matrices

    LOW_RANK = "low_rank"
    RANK_ONE = "rank_one"

    SYMMETRIC = "symmetric"  # 𝕊ₙ(R)
    SKEW_SYMMETRIC = "skew_symmetric"
    POSITIVE_DEFINITE = "positive_definite"  # 𝕊ₙ⁺(ℝ)
    NEGATIVE_DEFINITE = "negative_definite"  # 𝕊ₙ⁻(ℝ)
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"  # 𝕊ₙ⁺(ℝ) ∪ {0}
    NEGATIVE_SEMIDEFINITE = "negative_semidefinite"  # 𝕊ₙ⁻(ℝ) ∪ {0}

    # determinant-based
    SINGULAR = "singular"  # det=0
    INVERTIBLE = "invertible"  # GLₙ(R) (det≠0)
    POSITIVE_DETERMINANT = "positive_determinant"  # GLₙ⁺(R) (det>0)
    NEGATIVE_DETERMINANT = "negative_determinant"  # GLₙ⁻(R) (det<0)

    CONTRACTION = "contraction"  # ‖A‖₂ < 1
    SPECTRAL_NORMALIZED = "spectral_normalized"  # ‖A‖₂ = 1
    LIPSCHITZ_BOUNDED = "lipschitz_bounded"  # ‖A‖₂ ≤ C
    DIAGONALLY_DOMINANT = "diagonally_dominant"  # |Aᵢᵢ| ≥ ∑_{j≠i} |Aᵢⱼ| for all i

    NORMAL = "normal"
    ORTHOGONAL = "orthogonal"  # Oₙ(R)
    CAYLEY_ORTHOGONAL = "cayley_orthogonal"  # {Q ∈ SOₙ(n) ∣ -1 ∉ spec(Q)}
    SPECIAL_ORTHOGONAL = "special_orthogonal"  # SOₙ(R)
    PERMUTATION = "permutation"

    TRACELESS = "traceless"
    SYMPLECTIC = "symplectic"
    HAMILTONIAN = "hamiltonian"

    MASKED = "masked"
    DIAGONAL = "diagonal"
    TRIDIAGONAL = "tridiagonal"
    UPPER_TRIANGULAR = "upper_triangular"
    LOWER_TRIANGULAR = "lower_triangular"
    BANDED = "banded"

    ROW_STOCHASTIC = "row_stochastic"
    COLUMN_STOCHASTIC = "column_stochastic"
    DOUBLY_STOCHASTIC = "doubly_stochastic"


M = MatrixDomains  # temporary alias
MatrixDomains.KNOWN_EDGES = MappingProxyType({
    M.SQUARE: frozenset({M.RECTANGULAR}),
    M.EVEN_SQUARE: frozenset({M.SQUARE}),
    M.LOW_RANK: frozenset({M.RECTANGULAR}),
    M.RANK_ONE: frozenset({M.LOW_RANK}),
    M.SYMMETRIC: frozenset({M.SQUARE, M.NORMAL}),
    M.SKEW_SYMMETRIC: frozenset({M.SQUARE, M.NORMAL}),
    M.POSITIVE_DEFINITE: frozenset({M.SYMMETRIC, M.INVERTIBLE, M.POSITIVE_SEMIDEFINITE}),
    M.NEGATIVE_DEFINITE: frozenset({M.SYMMETRIC, M.INVERTIBLE, M.NEGATIVE_SEMIDEFINITE}),
    M.POSITIVE_SEMIDEFINITE: frozenset({M.SYMMETRIC}),
    M.NEGATIVE_SEMIDEFINITE: frozenset({M.SYMMETRIC}),
    M.SINGULAR: frozenset({M.SQUARE}),
    M.INVERTIBLE: frozenset({M.SQUARE}),
    M.POSITIVE_DETERMINANT: frozenset({M.INVERTIBLE}),
    M.NEGATIVE_DETERMINANT: frozenset({M.INVERTIBLE}),
    M.CONTRACTION: frozenset({M.RECTANGULAR}),
    M.SPECTRAL_NORMALIZED: frozenset({M.RECTANGULAR, M.LIPSCHITZ_BOUNDED}),
    M.LIPSCHITZ_BOUNDED: frozenset({M.RECTANGULAR}),
    M.DIAGONALLY_DOMINANT: frozenset({M.SQUARE}),
    M.NORMAL: frozenset({M.SQUARE}),
    M.ORTHOGONAL: frozenset({M.SQUARE, M.INVERTIBLE, M.NORMAL, M.SPECTRAL_NORMALIZED}),
    M.CAYLEY_ORTHOGONAL: frozenset({M.SPECIAL_ORTHOGONAL}),
    M.SPECIAL_ORTHOGONAL: frozenset({M.ORTHOGONAL, M.POSITIVE_DETERMINANT}),
    M.PERMUTATION: frozenset({M.ORTHOGONAL, M.DOUBLY_STOCHASTIC}),
    M.TRACELESS: frozenset({M.SQUARE}),
    M.SYMPLECTIC: frozenset({M.EVEN_SQUARE, M.INVERTIBLE, M.POSITIVE_DETERMINANT}),
    M.HAMILTONIAN: frozenset({M.EVEN_SQUARE, M.TRACELESS}),
    M.DIAGONAL: frozenset({M.SYMMETRIC, M.TRIDIAGONAL, M.UPPER_TRIANGULAR, M.LOWER_TRIANGULAR}),
    M.TRIDIAGONAL: frozenset({M.BANDED, M.SQUARE}),
    M.UPPER_TRIANGULAR: frozenset({M.SQUARE}),
    M.LOWER_TRIANGULAR: frozenset({M.SQUARE}),
    M.BANDED: frozenset({M.RECTANGULAR}),
    M.ROW_STOCHASTIC: frozenset({M.RECTANGULAR}),
    M.COLUMN_STOCHASTIC: frozenset({M.RECTANGULAR}),
    M.DOUBLY_STOCHASTIC: frozenset({M.ROW_STOCHASTIC, M.COLUMN_STOCHASTIC, M.SQUARE}),
})  # fmt: skip
del M  # remove alias
