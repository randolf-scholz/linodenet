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
    "Fallback",
]

from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import Final

import torch
from torch import Tensor

from .base import MatrixDomain, PosetEnum


@dataclass(frozen=True)
class Fallback(MatrixDomain):
    r"""Named placeholder for an otherwise unspecified matrix domain."""

    name: str

    def __contains__(self, item: Tensor, /) -> bool:
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

    def __contains__(self, item: Tensor, /) -> bool:
        if self.shape is None:
            return True
        return item.shape[-2:] == self.shape

    def __call__(self, rows: int | None = None, cols: int | None = None) -> Rectangular:
        return Rectangular(rows, cols)


@dataclass(frozen=True)
class Tall(Rectangular):
    r"""Domain of matrices with at least as many rows as columns."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rows is not None and self.cols is not None and self.rows < self.cols:
            raise ValueError("Tall matrices must satisfy rows >= cols.")

    def __contains__(self, item: Tensor, /) -> bool:
        return super().__contains__(item) and item.shape[-2] >= item.shape[-1]

    def __call__(self, rows: int | None = None, cols: int | None = None) -> Tall:
        return Tall(rows, cols)


@dataclass(frozen=True)
class Wide(Rectangular):
    r"""Domain of matrices with at least as many columns as rows."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rows is not None and self.cols is not None and self.cols < self.rows:
            raise ValueError("Wide matrices must satisfy cols >= rows.")

    def __contains__(self, item: Tensor, /) -> bool:
        return super().__contains__(item) and item.shape[-1] >= item.shape[-2]

    def __call__(self, rows: int | None = None, cols: int | None = None) -> Wide:
        return Wide(rows, cols)


@dataclass(frozen=True)
class ColumnOrthogonal(Tall):
    r"""Domain of tall matrices with orthonormal columns."""

    def __contains__(self, item: Tensor, /) -> bool:
        if not super().__contains__(item):
            return False
        cols = item.shape[-1]
        gram = item.mT @ item
        eye = torch.eye(cols, dtype=item.dtype, device=item.device)
        return bool(torch.allclose(gram, eye))

    def __call__(
        self, rows: int | None = None, cols: int | None = None
    ) -> ColumnOrthogonal:
        return ColumnOrthogonal(rows, cols)


@dataclass(frozen=True)
class RowOrthogonal(Wide):
    r"""Domain of wide matrices with orthonormal rows."""

    def __contains__(self, item: Tensor, /) -> bool:
        if not super().__contains__(item):
            return False
        rows = item.shape[-2]
        gram = item @ item.mT
        eye = torch.eye(rows, dtype=item.dtype, device=item.device)
        return bool(torch.allclose(gram, eye))

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

    def __contains__(self, item: Tensor, /) -> bool:
        if self.size is None:
            return item.shape[-1] == item.shape[-2]
        return item.shape[-2:] == self.shape

    def __call__(self, size: int) -> Square:
        return Square(size)


@dataclass(frozen=True)
class LowRank(Rectangular):
    r"""Domain of rectangular matrices with optional rank bound."""

    _: KW_ONLY
    rank: Final[int | None] = None

    def __contains__(self, item: Tensor, /) -> bool:
        raise NotImplementedError

    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        *,
        rank: int | None = None,
    ) -> LowRank:
        return LowRank(rows, cols, rank=rank)


@dataclass(frozen=True)
class Symmetric(Square):
    r"""Domain of symmetric square matrices."""

    def __contains__(self, item: Tensor, /) -> bool:
        raise NotImplementedError

    def __call__(self, size: int) -> Symmetric:
        return Symmetric(size)


@dataclass(frozen=True)
class SkewSymmetric(Square):
    r"""Domain of skew-symmetric square matrices."""

    def __contains__(self, item: Tensor, /) -> bool:
        raise NotImplementedError

    def __call__(self, size: int) -> SkewSymmetric:
        return SkewSymmetric(size)


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
    LOW_RANK = "low_rank"  # rank small relative to size
    RANK_ONE = "rank_one"

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

    def __contains__(self, item: Tensor, /) -> bool:
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
