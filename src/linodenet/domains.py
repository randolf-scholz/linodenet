r"""WORK IN PROGRESS.

Domains should allow:

1. checking membership of tensors
2. checking subset relations between domains
3. performing some basic operations (e.g. product of domains, union, intersection)
"""

__all__ = [
    "Domain",
    "Interval",
    "UnionOfIntervals",
    "ScalarDomains",
    "VectorDomains",
    "MatrixDomains",
]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from types import MappingProxyType
from typing import ClassVar, Final, Protocol, Self, overload

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


@dataclass(unsafe_hash=True)
class Interval(Domain):
    r"""A named tuple representing an interval."""

    lower: Final[float]
    upper: Final[float]
    lower_inclusive: Final[bool]
    upper_inclusive: Final[bool]

    @overload
    def __init__(self, s: str, /) -> None: ...

    @overload
    def __init__(self, interval: Interval, /) -> None: ...

    @overload
    def __init__(
        self,
        lower: float,
        upper: float,
        *,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> None: ...

    def __init__(
        self,
        lower: str | Interval | float,
        upper: float | None = None,
        *,
        lower_inclusive: bool | None = None,
        upper_inclusive: bool | None = None,
    ) -> None:
        if isinstance(lower, str | Interval):
            if (
                upper is not None
                or lower_inclusive is not None
                or upper_inclusive is not None
            ):
                raise TypeError("String interval constructor does not accept bounds.")
            interval = (
                lower if isinstance(lower, Interval) else type(self).from_string(lower)
            )
            self.lower = interval.lower
            self.upper = interval.upper
            self.lower_inclusive = interval.lower_inclusive
            self.upper_inclusive = interval.upper_inclusive

        else:
            if upper is None or lower_inclusive is None or upper_inclusive is None:
                raise TypeError(
                    "Expected upper and inclusivity flags for numeric bounds."
                )

            self.lower = lower  # type: ignore[misc]
            self.upper = upper  # type: ignore[misc]
            self.lower_inclusive = lower_inclusive  # type: ignore[misc]
            self.upper_inclusive = upper_inclusive  # type: ignore[misc]

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

    def __add__(self, other: float, /) -> Self:
        return type(self)(
            self.lower + other,
            self.upper + other,
            lower_inclusive=self.lower_inclusive,
            upper_inclusive=self.upper_inclusive,
        )

    def __sub__(self, other: float, /) -> Self:
        return self + (-other)

    def __mul__(self, other: float, /) -> Self:
        if other == 0:
            return type(self)(
                0.0,
                0.0,
                lower_inclusive=True,
                upper_inclusive=True,
            )
        if other > 0:
            return type(self)(
                self.lower * other,
                self.upper * other,
                lower_inclusive=self.lower_inclusive,
                upper_inclusive=self.upper_inclusive,
            )
        return type(self)(
            self.upper * other,
            self.lower * other,
            lower_inclusive=self.upper_inclusive,
            upper_inclusive=self.lower_inclusive,
        )

    @staticmethod
    def _coerce_interval(other: object, /) -> Interval | None:
        match other:
            case Interval():
                return other
            case str():
                return Interval.from_string(other)
            case _:
                return None

    def __eq__(self, other: object, /) -> bool:
        if (other_interval := self._coerce_interval(other)) is None:
            return NotImplemented
        return hash(self) == hash(other_interval)

    def __le__(self, other: object, /) -> bool:
        if isinstance(other, UnionOfIntervals):
            return any(self <= interval for interval in other.intervals)
        if isinstance(other, str):
            return self <= UnionOfIntervals.from_string(other)
        if (other_interval := self._coerce_interval(other)) is None:
            return NotImplemented

        lower_ok = self.lower > other_interval.lower or (
            self.lower == other_interval.lower
            and (other_interval.lower_inclusive or not self.lower_inclusive)
        )
        upper_ok = self.upper < other_interval.upper or (
            self.upper == other_interval.upper
            and (other_interval.upper_inclusive or not self.upper_inclusive)
        )
        return lower_ok and upper_ok

    def __ge__(self, other: object, /) -> bool:
        if isinstance(other, UnionOfIntervals):
            return all(interval <= self for interval in other.intervals)
        if isinstance(other, str):
            return self >= UnionOfIntervals.from_string(other)
        if (other_interval := self._coerce_interval(other)) is None:
            return NotImplemented
        return other_interval <= self

    def __str__(self) -> str:
        lower_bracket = "[" if self.lower_inclusive else "("
        upper_bracket = "]" if self.upper_inclusive else ")"
        lower = format(self.lower, "g")
        upper = format(self.upper, "g")
        return f"{lower_bracket}{lower}, {upper}{upper_bracket}"

    def __repr__(self) -> str:
        return f"Interval('{self!s}')"


@dataclass(init=False)
class UnionOfIntervals(Domain):
    r"""A finite union of intervals with automatic simplification."""

    intervals: Final[tuple[Interval, ...]]

    def __init__(self, *intervals: Interval | str) -> None:
        if not intervals:
            raise ValueError("Expected at least one interval.")
        self.intervals = self._merge_intervals(Interval(spec) for spec in intervals)

    @classmethod
    def from_string(cls, s: str, /) -> Self:
        r"""Create a union of intervals from a `|`-separated string."""
        parts = [part.strip() for part in s.split("|")]
        if any(not part for part in parts):
            raise ValueError(f"Invalid union of intervals string: {s}")
        return cls(*(Interval.from_string(part) for part in parts))

    @staticmethod
    def _coerce_union(other: object, /) -> UnionOfIntervals | None:
        match other:
            case UnionOfIntervals():
                return other
            case Interval():
                return UnionOfIntervals(other)
            case str():
                return UnionOfIntervals.from_string(other)
            case _:
                return None

    @staticmethod
    def _merge_intervals(intervals: Iterable[Interval], /) -> tuple[Interval, ...]:
        ordered = sorted(intervals, key=lambda i: (i.lower, not i.lower_inclusive))

        merged: list[Interval] = []
        for interval in ordered:
            if not merged:
                merged.append(interval)
                continue

            current = merged[-1]
            if not UnionOfIntervals._touch_or_overlap(current, interval):
                merged.append(interval)
                continue

            merged[-1] = Interval(
                current.lower,
                interval.upper,
                lower_inclusive=current.lower_inclusive,
                upper_inclusive=interval.upper_inclusive,
            )

        return tuple(merged)

    @staticmethod
    def _touch_or_overlap(left: Interval, right: Interval, /) -> bool:
        if right.lower < left.upper:
            return True
        if right.lower > left.upper:
            return False
        return left.upper_inclusive or right.lower_inclusive

    def __contains__(self, item: Tensor, /) -> Tensor:
        mask = self.intervals[0].__contains__(item)
        for interval in self.intervals[1:]:
            mask = mask | interval.__contains__(item)
        return mask

    def __add__(self, other: float, /) -> Self:
        return type(self)(*(interval + other for interval in self.intervals))

    def __sub__(self, other: float, /) -> Self:
        return self + (-other)

    def __mul__(self, other: float, /) -> Self:
        return type(self)(*(interval * other for interval in self.intervals))

    def __le__(self, other: object, /) -> bool:
        if (other_union := self._coerce_union(other)) is None:
            return NotImplemented

        return all(
            any(interval <= other_interval for other_interval in other_union.intervals)
            for interval in self.intervals
        )

    def __str__(self) -> str:
        return " | ".join(str(interval) for interval in self.intervals)

    def __repr__(self) -> str:
        return f"UnionOfIntervals('{self!s}')"


class ScalarDomains(_PosetEnum):
    r"""Enumeration of some scalar domains."""

    EXTENDED_LINE = Interval.from_string("[-inf, inf]")
    REAL_LINE = Interval.from_string("(-inf, inf)")
    POSITIVE_REALS = Interval.from_string("(0, inf)")
    NEGATIVE_REALS = Interval.from_string("(-inf, 0)")
    NONNEGATIVE_REALS = Interval.from_string("[0, inf)")
    NONPOSITIVE_REALS = Interval.from_string("(-inf, 0]")
    NONZERO = UnionOfIntervals.from_string("(-inf, 0) | (0, inf)")

    UNIT_INTERVAL = Interval.from_string("[0, 1]")
    OPEN_UNIT_INTERVAL = Interval.from_string("(0, 1)")

    @property
    def domain(self) -> Domain:
        return self.value

    def __contains__(self, item: Tensor, /) -> Tensor:
        return self.domain.__contains__(item)


S = ScalarDomains  # temporary alias
ScalarDomains.KNOWN_EDGES = MappingProxyType({
    S.REAL_LINE: frozenset({S.EXTENDED_LINE}),
    S.UNIT_INTERVAL: frozenset({S.NONNEGATIVE_REALS, S.REAL_LINE}),
    S.OPEN_UNIT_INTERVAL: frozenset({S.UNIT_INTERVAL, S.POSITIVE_REALS}),
    S.POSITIVE_REALS: frozenset({S.NONNEGATIVE_REALS, S.NONZERO, S.REAL_LINE}),
    S.NONNEGATIVE_REALS: frozenset({S.REAL_LINE}),
    S.NEGATIVE_REALS: frozenset({S.NONPOSITIVE_REALS, S.NONZERO, S.REAL_LINE}),
    S.NONPOSITIVE_REALS: frozenset({S.REAL_LINE}),
    S.NONZERO: frozenset({S.REAL_LINE, S.EXTENDED_LINE}),
})  # fmt: skip
del S  # remove alias


class VectorDomains(_PosetEnum):
    r"""Enumeration of some vector domains."""

    REAL = "real"
    COMPLEX = "complex"
    BOOLEAN = "boolean"

    SPARSE = "sparse"  # xᵢ=0 for many i
    ONE_HOT = "one-hot"  # xᵢ=1, xⱼ=0 for j≠i

    ZERO_MEAN = "zero-mean"
    STANDARDIZED = "standardized"  # zero-mean, unit variance

    UNIT_VECTOR = "unit_sphere"  # ‖x‖₂ = 1
    STOCHASTIC = "stochastic"  # sum(x) = 1, x ≥ 0

    NONZERO = "nonzero"  # x ≠ 0
    POSITIVE = "positive"  # xᵢ > 0
    NEGATIVE = "negative"  # xᵢ < 0
    NONNEGATIVE = "nonnegative"  # xᵢ ≥ 0
    NONPOSITIVE = "nonpositive"  # xᵢ ≤ 0


V = VectorDomains  # temporary alias
VectorDomains.KNOWN_EDGES = MappingProxyType({
    V.REAL: frozenset({V.COMPLEX}),
    V.BOOLEAN: frozenset({V.REAL, V.NONNEGATIVE}),
    V.ONE_HOT: frozenset({V.BOOLEAN, V.STOCHASTIC, V.UNIT_VECTOR, V.SPARSE}),
    V.STANDARDIZED: frozenset({V.ZERO_MEAN, V.NONZERO}),
    V.UNIT_VECTOR: frozenset({V.NONZERO}),
    V.STOCHASTIC: frozenset({V.REAL, V.NONNEGATIVE, V.NONZERO}),
    V.POSITIVE: frozenset({V.REAL, V.NONNEGATIVE, V.NONZERO}),
    V.NEGATIVE: frozenset({V.REAL, V.NONPOSITIVE, V.NONZERO}),
    V.NONNEGATIVE: frozenset({V.REAL}),
    V.NONPOSITIVE: frozenset({V.REAL}),
})  # fmt: skip
del V  # remove alias


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
