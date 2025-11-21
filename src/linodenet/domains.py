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

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

from torch import Tensor


class Domain(Protocol):
    r"""Protocol for Domains."""

    def __contains__(self, item: Tensor, /) -> Tensor: ...

    def __le__(self, other: "Domain", /) -> bool: ...
    def __gt__(self, other: "Domain", /) -> bool: ...
    def __ge__(self, other: "Domain", /) -> bool: ...
    def __lt__(self, other: "Domain", /) -> bool: ...


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
    def from_string(cls, s: str) -> "Interval":
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

    GENERAL = "general"
    SPARSE = "sparse"

    UNIT_SPHERE = "unit_sphere"  # ‖x‖₂ = 1
    NONZERO = "nonzero"  # x ≠ 0

    NONNEGATIVE = "nonnegative"  # x ≥ 0
    POSITIVE = "positive"  # x > 0
    NONPOSITIVE = "nonpositive"  # x ≤ 0
    NEGATIVE = "negative"  # x < 0

    STOCHASTIC = "stochastic"  # sum(x) = 1, x ≥ 0
    PROBABILITY_SIMPLEX = "probability_simplex"  # sum(x) = 1, x ≥ 0


class MatrixDomains(StrEnum):
    r"""Enumeration of some matrix domains."""

    GENERAL = "general"
    LOW_RANK = "low_rank"

    SQUARE = "square"  # n x n matrices
    EVEN_SQUARE = "even_square"  # 2n x 2n matrices

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

    NORMAL = "normal"
    ORTHOGONAL = "orthogonal"  # Oₙ(R)
    SPECIAL_ORTHOGONAL = "special_orthogonal"  # SOₙ(R)
    PERMUTATION = "permutation"

    TRACELESS = "traceless"
    SYMPLECTIC = "symplectic"
    HAMILTONIAN = "hamiltonian"

    MASKED = "masked"
    DIAGONAL = "diagonal"
    UPPER_TRIANGULAR = "upper_triangular"
    LOWER_TRIANGULAR = "lower_triangular"
    BANDED = "banded"

    STOCHASTIC = "stochastic"
    DOUBLY_STOCHASTIC = "doubly_stochastic"
