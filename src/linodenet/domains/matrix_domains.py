r"""Matrix-specific domain primitives and partial-order labels."""

__all__ = [
    "MatrixDomains",
    # Classes
    "Square",
    "Rectangular",
    "Tall",
    "Wide",
    "ColumnOrthogonal",
    "RowOrthogonal",
    "Symmetric",
    "SkewSymmetric",
    "LowRank",
    "LowRankSquare",
    "LowRankSymmetric",
    "LowRankSkewSymmetric",
    "Fallback",
]

from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import Final

import torch
from torch import Tensor

from .base import MatrixDomain, PosetEnum
from .matrix_tests import (
    is_low_rank,
    is_low_rank_skew_symmetric,
    is_low_rank_square,
    is_low_rank_symmetric,
    is_skew_symmetric,
    is_symmetric,
)


@dataclass(frozen=True)
class Fallback(MatrixDomain):
    r"""Named placeholder for an otherwise unspecified matrix domain."""

    name: str

    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class Rectangular(MatrixDomain):
    r"""Domain of rectangular matrices with optional fixed shape."""

    rows: Final[int | None] = None  # pyright: ignore[reportIncompatibleMethodOverride]
    cols: Final[int | None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        if (self.rows is None) ^ (self.cols is None):
            raise ValueError("Must specify both rows and cols, or neither.")

    @property
    def shape(self) -> tuple[int, int] | None:
        if self.rows is None or self.cols is None:
            return None
        return self.rows, self.cols

    def check(self, value: Tensor, /) -> Tensor:
        *batch_shape, m, n = value.shape
        if self.shape is None:
            return value.new_full(batch_shape, True, dtype=torch.bool)
        return value.new_full(batch_shape, self.shape == (m, n), dtype=torch.bool)

    def __call__(self, rows: int | None = None, cols: int | None = None) -> Rectangular:
        return Rectangular(rows, cols)


@dataclass(frozen=True)
class Tall(Rectangular):
    r"""Domain of matrices with at least as many rows as columns."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rows is not None and self.cols is not None and self.rows < self.cols:
            raise ValueError("Tall matrices must satisfy rows >= cols.")

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & value.new_full(
            value.shape[:-2], value.shape[-2] >= value.shape[-1], dtype=torch.bool
        )

    def __call__(self, rows: int | None = None, cols: int | None = None) -> Tall:
        return Tall(rows, cols)


@dataclass(frozen=True)
class Wide(Rectangular):
    r"""Domain of matrices with at least as many columns as rows."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rows is not None and self.cols is not None and self.cols < self.rows:
            raise ValueError("Wide matrices must satisfy cols >= rows.")

    def check(self, value: Tensor, /) -> Tensor:
        return super().check(value) & value.new_full(
            value.shape[:-2], value.shape[-1] >= value.shape[-2], dtype=torch.bool
        )

    def __call__(self, rows: int | None = None, cols: int | None = None) -> Wide:
        return Wide(rows, cols)


@dataclass(frozen=True)
class ColumnOrthogonal(Tall):
    r"""Domain of tall matrices with orthonormal columns."""

    def check(self, value: Tensor, /) -> Tensor:
        shape_ok = super().check(value)
        if not bool(shape_ok.all()):
            return shape_ok & False

        cols = value.shape[-1]
        gram = value.mT @ value
        eye = torch.eye(cols, dtype=value.dtype, device=value.device)
        return shape_ok & torch.isclose(gram, eye).all(dim=(-2, -1))

    def __call__(
        self, rows: int | None = None, cols: int | None = None
    ) -> ColumnOrthogonal:
        return ColumnOrthogonal(rows, cols)


@dataclass(frozen=True)
class RowOrthogonal(Wide):
    r"""Domain of wide matrices with orthonormal rows."""

    def check(self, value: Tensor, /) -> Tensor:
        shape_ok = super().check(value)
        if not bool(shape_ok.all()):
            return shape_ok & False

        rows = value.shape[-2]
        gram = value @ value.mT
        eye = torch.eye(rows, dtype=value.dtype, device=value.device)
        return shape_ok & torch.isclose(gram, eye).all(dim=(-2, -1))

    def __call__(
        self, rows: int | None = None, cols: int | None = None
    ) -> RowOrthogonal:
        return RowOrthogonal(rows, cols)


@dataclass(frozen=True)
class Square(MatrixDomain):
    r"""Domain of square matrices with optional fixed size."""

    size: Final[int | None] = None

    @property
    def rows(self) -> int | None:
        return self.size

    @property
    def cols(self) -> int | None:
        return self.size

    @property
    def shape(self) -> tuple[int, int] | None:
        if self.size is None:
            return None
        return self.size, self.size

    def check(self, value: Tensor, /) -> Tensor:
        *batch_shape, m, n = value.shape
        if self.size is None:
            return value.new_full(batch_shape, m == n, dtype=torch.bool)
        return value.new_full(batch_shape, (m, n) == self.shape, dtype=torch.bool)

    def __call__(self, size: int) -> Square:
        return Square(size)


@dataclass(frozen=True)
class LowRank(Rectangular):
    r"""Domain of rectangular matrices with optional rank bound."""

    _: KW_ONLY
    rank: Final[int | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        shape_ok = super().check(value)
        if self.rank is None or not bool(shape_ok.all()):
            return shape_ok
        return is_low_rank(value, self.rank, shape=self.shape)

    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        *,
        rank: int | None = None,
    ) -> LowRank:
        return LowRank(rows, cols, rank=rank)


@dataclass(frozen=True)
class LowRankSquare(Square):
    r"""Domain of square matrices with optional rank bound."""

    _: KW_ONLY
    rank: Final[int | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        if self.rank is None:
            return super().check(value)
        return is_low_rank_square(value, self.rank, size=self.size)

    def __call__(
        self, size: int | None = None, *, rank: int | None = None
    ) -> LowRankSquare:
        return LowRankSquare(size, rank=rank)


@dataclass(frozen=True)
class Symmetric(Square):
    r"""Domain of symmetric square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return is_symmetric(value, size=self.size)

    def __call__(self, size: int) -> Symmetric:
        return Symmetric(size)


@dataclass(frozen=True)
class SkewSymmetric(Square):
    r"""Domain of skew-symmetric square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return is_skew_symmetric(value, size=self.size)

    def __call__(self, size: int) -> SkewSymmetric:
        return SkewSymmetric(size)


@dataclass(frozen=True)
class LowRankSymmetric(Square):
    r"""Domain of matrices of the form $UVᵀ + VUᵀ$ with optional factor rank bound."""

    _: KW_ONLY
    rank: Final[int | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        if self.rank is None:
            return is_symmetric(value, size=self.size)
        return is_low_rank_symmetric(value, self.rank, size=self.size)

    def __call__(
        self, size: int | None = None, *, rank: int | None = None
    ) -> LowRankSymmetric:
        return LowRankSymmetric(size, rank=rank)


@dataclass(frozen=True)
class LowRankSkewSymmetric(Square):
    r"""Domain of matrices of the form $UVᵀ - VUᵀ$ with optional factor rank bound."""

    _: KW_ONLY
    rank: Final[int | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        if self.rank is None:
            return is_skew_symmetric(value, size=self.size)
        return is_low_rank_skew_symmetric(value, self.rank, size=self.size)

    def __call__(
        self, size: int | None = None, *, rank: int | None = None
    ) -> LowRankSkewSymmetric:
        return LowRankSkewSymmetric(size, rank=rank)


class MatrixDomains(PosetEnum):
    r"""Enumeration of some matrix domains."""

    ANY = "any"  # top node
    NONE = "none"  # bottom node

    BOOLEAN = "boolean"  # only 0 and 1 entries
    SPARSE = "sparse"  # many 0 entries
    MASKED = "masked"  # X⊙M = X for some mask M

    RECTANGULAR = "rectangular"  # m × n matrices
    TALL = "tall"  # m × n matrices with m ≥ n
    WIDE = "wide"  # m × n matrices with m ≤ n
    COLUMN_ORTHOGONAL = "column_orthogonal"  # m × n matrices with QᵀQ = 𝕀ₙ
    ROW_ORTHOGONAL = "row_orthogonal"  # m × n matrices with QQᵀ = 𝕀ₘ
    SQUARE = "square"  # n × n matrices
    EVEN_SQUARE = "even_square"  # 2n × 2n matrices

    # specific matrices
    ZERO = "zero"  # the zero matrix
    IDENTITY = "identity"  # the identity matrix
    EYE = "identity"  # alias
    STANDARD_SYMPLECTIC = "standard_symplectic"  # [0, 𝕀; -𝕀, 0]

    # rank
    LOW_RANK = "low_rank"  # UVᵀ
    LOW_RANK_SQUARE = "low_rank_square"
    LOW_RANK_SYMMETRIC = "low_rank_symmetric"  # UVᵀ + VUᵀ
    LOW_RANK_SKEW_SYMMETRIC = "low_rank_skew_symmetric"  # UVᵀ - VUᵀ
    RANK_ONE = "rank_one"  # uvᵀ

    # determinant-based
    SINGULAR = "singular"  # det=0
    LEFT_INVERTIBLE = "left_invertible"  # full column rank, admits L with LA = 𝕀
    RIGHT_INVERTIBLE = "right_invertible"  # full row rank, admits R with AR = 𝕀
    INVERTIBLE = "invertible"  # GLₙ(R) (det≠0)
    LOWER_INVERTIBLE = "lower_invertible"
    UPPER_INVERTIBLE = "upper_invertible"
    CHOLESKY_FACTOR = "cholesky_factor"
    UNIT_DETERMINANT = "unit_determinant"  # SLₙ(R) (det=1)
    GENERAL_LINEAR = "invertible"  # alias
    SPECIAL_LINEAR = "unit_determinant"  # alias
    POSITIVE_DETERMINANT = "positive_determinant"  # GLₙ⁺(R) (det>0)
    NEGATIVE_DETERMINANT = "negative_determinant"  # GLₙ⁻(R) (det<0)

    # symmetry / entry based
    SYMMETRIC = "symmetric"  # 𝕊ₙ(R)
    SKEW_SYMMETRIC = "skew_symmetric"
    DIAGONAL = "diagonal"
    POSITIVE_DIAGONAL_ENTRIES = "positive_diagonal_entries"
    NEGATIVE_DIAGONAL_ENTRIES = "negative_diagonal_entries"
    ZERO_DIAGONAL_ENTRIES = "zero_diagonal"
    ZERO_DIAGONAL = "zero_diagonal"
    TRIDIAGONAL = "tridiagonal"
    BANDED = "banded"
    TOEPLITZ = "toeplitz"  # constant along diagonals
    HANKEL = "hankel"  # constant along anti-diagonals
    CIRCULANT = "circulant"  # constant along diagonals, wrap around

    # eigenvalues
    POSITIVE_DEFINITE = "positive_definite"  # 𝕊ₙ⁺(ℝ)
    NEGATIVE_DEFINITE = "negative_definite"  # 𝕊ₙ⁻(ℝ)
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"  # 𝕊ₙ⁺(ℝ) ∪ {0}
    NEGATIVE_SEMIDEFINITE = "negative_semidefinite"  # 𝕊ₙ⁻(ℝ) ∪ {0}

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
    SYMPLECTIC = "symplectic"  # 2n×2n with AᵀJA = J for J=[0, I;-I, 0]
    HAMILTONIAN = "hamiltonian"  # 2n×2n with (JA)ᵀ = JA for J=[0, I;-I, 0]

    TRIANGULAR = "triangular"  # lower or upper
    UPPER_TRIANGULAR = "upper_triangular"
    LOWER_TRIANGULAR = "lower_triangular"

    ROW_STOCHASTIC = "row_stochastic"
    COLUMN_STOCHASTIC = "column_stochastic"
    DOUBLY_STOCHASTIC = "doubly_stochastic"

    IDEMPOTENT = "idempotent"  # Aᵏ = A for some k≥2
    PROJECTION = "projection"  # A² = A
    NILPOTENT = "nilpotent"  # Aᵏ = 0 for some k≥2

    EFFICIENTLY_INVERTIBLE = "efficient_invertible"

    HADAMARD = "hadamard"  # entries ±1, HHᵀ=n𝕀

    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError


M = MatrixDomains  # temporary alias
MatrixDomains.KNOWN_MEETS = (
    (M.NONE, M.CONTRACTION & M.SPECTRAL_NORMALIZED),
    (M.NONE, M.INVERTIBLE & M.SINGULAR),
    (M.NONE, M.NEGATIVE_DEFINITE & M.POSITIVE_DEFINITE),
    (M.NONE, M.NEGATIVE_DEFINITE & M.POSITIVE_SEMIDEFINITE),
    (M.NONE, M.NEGATIVE_DETERMINANT & M.POSITIVE_DETERMINANT),
    (M.NONE, M.NEGATIVE_DETERMINANT & M.UNIT_DETERMINANT),
    (M.NONE, M.NEGATIVE_SEMIDEFINITE & M.POSITIVE_DEFINITE),
    (M.ZERO, M.DIAGONAL & M.ZERO_DIAGONAL),
    (M.ZERO, M.NEGATIVE_SEMIDEFINITE & M.POSITIVE_SEMIDEFINITE),
    (M.ZERO, M.NEGATIVE_SEMIDEFINITE & M.SKEW_SYMMETRIC),
    (M.ZERO, M.POSITIVE_SEMIDEFINITE & M.SKEW_SYMMETRIC),
    (M.ZERO, M.SYMMETRIC & M.SKEW_SYMMETRIC),
    (M.EYE, M.POSITIVE_DEFINITE & M.ORTHOGONAL),
    (M.EYE, M.DIAGONAL & M.PERMUTATION),
    (M.DIAGONAL, M.LOWER_TRIANGULAR & M.UPPER_TRIANGULAR),
    (M.CHOLESKY_FACTOR, M.LOWER_INVERTIBLE & M.POSITIVE_DIAGONAL_ENTRIES),
    (M.DOUBLY_STOCHASTIC, M.SQUARE & M.ROW_STOCHASTIC & M.COLUMN_STOCHASTIC),
    (M.INVERTIBLE, M.LEFT_INVERTIBLE & M.RIGHT_INVERTIBLE),
    (M.LOWER_INVERTIBLE, M.LOWER_TRIANGULAR & M.INVERTIBLE),
    (M.LOW_RANK_SKEW_SYMMETRIC, M.LOW_RANK_SQUARE & M.SKEW_SYMMETRIC),
    (M.LOW_RANK_SQUARE, M.LOW_RANK & M.SQUARE),
    (M.LOW_RANK_SYMMETRIC, M.LOW_RANK_SQUARE & M.SYMMETRIC),
    (M.NEGATIVE_DEFINITE, M.NEGATIVE_SEMIDEFINITE & M.INVERTIBLE),
    (M.ORTHOGONAL, M.COLUMN_ORTHOGONAL & M.ROW_ORTHOGONAL),
    (M.PERMUTATION, M.ORTHOGONAL & M.DOUBLY_STOCHASTIC),
    (M.POSITIVE_DEFINITE, M.POSITIVE_SEMIDEFINITE & M.INVERTIBLE),
    (M.SPECIAL_ORTHOGONAL, M.ORTHOGONAL & M.UNIT_DETERMINANT),
    (M.SQUARE, M.TALL & M.WIDE),
    (M.UPPER_INVERTIBLE, M.UPPER_TRIANGULAR & M.INVERTIBLE),
)
MatrixDomains.KNOWN_SUPERTYPES = MappingProxyType({
    M.BANDED: frozenset({M.RECTANGULAR}),
    M.CAYLEY_ORTHOGONAL: frozenset({M.SPECIAL_ORTHOGONAL}),
    M.CIRCULANT: frozenset({M.TOEPLITZ, M.SQUARE}),
    M.COLUMN_ORTHOGONAL: frozenset({M.LEFT_INVERTIBLE, M.SPECTRAL_NORMALIZED}),
    M.COLUMN_STOCHASTIC: frozenset({M.RECTANGULAR}),
    M.CONTRACTION: frozenset({M.LIPSCHITZ_BOUNDED}),
    M.DIAGONAL: frozenset({M.SYMMETRIC, M.TRIDIAGONAL}),
    M.DIAGONALLY_DOMINANT: frozenset({M.SQUARE}),
    M.EVEN_SQUARE: frozenset({M.SQUARE}),
    M.HAMILTONIAN: frozenset({M.EVEN_SQUARE, M.TRACELESS}),
    M.HANKEL: frozenset({M.RECTANGULAR}),
    M.LEFT_INVERTIBLE: frozenset({M.TALL}),
    M.LIPSCHITZ_BOUNDED: frozenset({M.RECTANGULAR}),
    M.LOWER_TRIANGULAR: frozenset({M.TRIANGULAR}),
    M.LOW_RANK: frozenset({M.RECTANGULAR}),
    M.NEGATIVE_DETERMINANT: frozenset({M.INVERTIBLE}),
    M.NEGATIVE_DIAGONAL_ENTRIES: frozenset({M.RECTANGULAR}),
    M.NEGATIVE_SEMIDEFINITE: frozenset({M.SYMMETRIC}),
    M.NORMAL: frozenset({M.SQUARE}),
    M.ORTHOGONAL: frozenset({M.NORMAL}),
    M.PERMUTATION: frozenset({M.SPARSE}),
    M.POSITIVE_DETERMINANT: frozenset({M.INVERTIBLE}),
    M.POSITIVE_DIAGONAL_ENTRIES: frozenset({M.RECTANGULAR}),
    M.POSITIVE_SEMIDEFINITE: frozenset({M.SYMMETRIC}),
    M.RANK_ONE: frozenset({M.LOW_RANK}),
    M.RIGHT_INVERTIBLE: frozenset({M.WIDE}),
    M.ROW_ORTHOGONAL: frozenset({M.RIGHT_INVERTIBLE, M.SPECTRAL_NORMALIZED}),
    M.ROW_STOCHASTIC: frozenset({M.RECTANGULAR}),
    M.SINGULAR: frozenset({M.SQUARE}),
    M.SKEW_SYMMETRIC: frozenset({M.SQUARE, M.NORMAL, M.ZERO_DIAGONAL}),
    M.SPARSE: frozenset({M.BOOLEAN}),
    M.SPECTRAL_NORMALIZED: frozenset({M.RECTANGULAR, M.LIPSCHITZ_BOUNDED}),
    M.STANDARD_SYMPLECTIC: frozenset({M.SYMPLECTIC, M.SKEW_SYMMETRIC}),
    M.SYMMETRIC: frozenset({M.SQUARE, M.NORMAL}),
    M.SYMPLECTIC: frozenset({M.EVEN_SQUARE, M.UNIT_DETERMINANT}),
    M.TALL: frozenset({M.RECTANGULAR}),
    M.TOEPLITZ: frozenset({M.RECTANGULAR}),
    M.TRACELESS: frozenset({M.SQUARE}),
    M.TRIDIAGONAL: frozenset({M.BANDED, M.SQUARE}),
    M.UNIT_DETERMINANT: frozenset({M.POSITIVE_DETERMINANT}),
    M.UPPER_TRIANGULAR: frozenset({M.TRIANGULAR}),
    M.WIDE: frozenset({M.RECTANGULAR}),
    M.ZERO: frozenset({M.SPARSE}),
    M.ZERO_DIAGONAL: frozenset({M.TRACELESS}),
})  # fmt: skip
MatrixDomains.KNOWN_SUBTYPES = MappingProxyType({
    M.EFFICIENTLY_INVERTIBLE: frozenset({
        M.SPARSE, M.PERMUTATION, M.ORTHOGONAL,
        M.TRIANGULAR, M.DIAGONAL, M.TRIDIAGONAL,
    }),
    M.SQUARE: frozenset({M.ROW_STOCHASTIC & M.COLUMN_STOCHASTIC}),
})  # fmt: skip
del M  # remove alias
