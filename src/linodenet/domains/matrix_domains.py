r"""Matrix-specific domain primitives and partial-order labels."""

__all__ = [
    "MatrixDomains",
    "Matrix",
    "Placeholder",
    # Classes
    "Empty",
    "Square",
    "EvenSquare",
    "Rectangular",
    "Boolean",
    "Tall",
    "Wide",
    "RankOne",
    "ColumnOrthogonal",
    "RowOrthogonal",
    "RowStochastic",
    "ColumnStochastic",
    "DoublyStochastic",
    "RowCentered",
    "ColumnCentered",
    "DoublyCentered",
    "LeftInvertible",
    "RightInvertible",
    "Symmetric",
    "SkewSymmetric",
    "Normal",
    "Orthogonal",
    "SpecialOrthogonal",
    "Projection",
    "OrthogonalProjection",
    "LowRank",
    "LowRankSquare",
    "LowRankSymmetric",
    "LowRankSkewSymmetric",
    "PositiveSemidefinite",
    "PositiveDefinite",
    "NegativeDiagonal",
    "PositiveDiagonal",
    "PositiveScalarMatrix",
    "NegativeSemidefinite",
    "NegativeDefinite",
    "Traceless",
    "Symplectic",
    "Hamiltonian",
    "Diagonal",
    "Triangular",
    "LowerTriangular",
    "UpperTriangular",
    "Tridiagonal",
    "BlockDiagonal",
    "Toeplitz",
    "Circulant",
    "Banded",
    "Masked",
    "SpectralNormalized",
    "LipschitzBounded",
    "Contraction",
    "DiagonallyDominant",
    "ForwardStable",
    "BackwardStable",
    "Sparse",
    "Zero",
    "Ones",
    "OneHot",
    "Identity",
    "Permutation",
]

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final, Self, overload

import torch
from torch import Tensor

from . import MatrixDomain, matrix_tests as tests
from .base import Indeterminate, PosetEnum


@dataclass(frozen=True)
class Matrix(MatrixDomain):
    r"""Matrix domain with optional admissible values."""

    rows: Final[int | None] = None  # pyright: ignore[reportIncompatibleMethodOverride]
    cols: Final[int | None] = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __post_init__(self) -> None:
        if (self.rows is None) ^ (self.cols is None):
            raise ValueError("Must specify both rows and cols, or neither.")

    def check(self, value: Tensor, /) -> Tensor:
        *batch_shape, m, n = value.shape
        if self.shape is None:
            return value.new_full(batch_shape, True, dtype=torch.bool)
        return value.new_full(batch_shape, self.shape == (m, n), dtype=torch.bool)

    @overload
    def __call__(self, /) -> Self: ...
    @overload
    def __call__(self, rows: int, cols: int, /) -> Self: ...
    def __call__(self, rows: int | None = None, cols: int | None = None, /) -> Self:
        return self.__class__(rows, cols)


@dataclass(frozen=True)
class Empty(Matrix):
    r"""Domain of matrices with no admissible values."""

    def check(self, value: Tensor, /) -> Tensor:
        return value.new_full(value.shape[:-2], False, dtype=torch.bool)


@dataclass(frozen=True)
class Placeholder(Matrix):
    r"""Named placeholder for an otherwise unspecified matrix domain."""

    _: KW_ONLY

    name: str

    @overload
    def __call__(self, /) -> Self: ...
    @overload
    def __call__(self, rows: int, cols: int, /) -> Self: ...
    def __call__(self, rows: int | None = None, cols: int | None = None, /) -> Self:
        return self.__class__(rows, cols, name=self.name)

    def check(self, value: Tensor, /) -> Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class Rectangular(Matrix):
    r"""Domain of rectangular matrices with optional fixed shape."""


@dataclass(frozen=True)
class Square(Rectangular):
    r"""Domain of square matrices with optional fixed size."""

    rows: Final[int | None] = field(default=None, repr=False)  # type: ignore[misc]
    cols: Final[int | None] = field(default=None, repr=False)  # type: ignore[misc]

    size: Final[int | None] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.cols is None:
            object.__setattr__(self, "cols", self.rows)
            object.__setattr__(self, "size", self.rows)
        elif self.rows is None:  # called as (None, int)
            raise ValueError("Expected size")
        elif self.rows != self.cols:
            raise ValueError("Square matrices must satisfy rows=cols.")

    @property
    def shape(self) -> tuple[int, int] | None:
        if self.size is None:
            return None
        return self.size, self.size

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_square(value, shape=self.shape)

    @overload
    def __call__(self, size: int | None = None, /) -> Self: ...
    @overload
    def __call__(self, rows: int, cols: int, /) -> Self: ...
    def __call__(self, rows: int | None = None, cols: int | None = None, /) -> Self:
        return self.__class__(rows, cols)


@dataclass(frozen=True)
class EvenSquare(Square):
    r"""Domain of square matrices with even size."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.size is not None and self.size % 2 != 0:
            raise ValueError("Even square matrices must have even size.")

    def check(self, value: Tensor, /) -> Tensor:
        shape_ok = super().check(value)
        return shape_ok & value.new_full(
            value.shape[:-2], value.shape[-2] % 2 == 0, dtype=torch.bool
        )


@dataclass(frozen=True)
class Tall(Rectangular):
    r"""Domain of matrices with at least as many rows as columns."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rows is not None and self.cols is not None and self.rows < self.cols:
            raise ValueError("Tall matrices must satisfy rows >= cols.")

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_tall(value, shape=self.shape)


@dataclass(frozen=True)
class Wide(Rectangular):
    r"""Domain of matrices with at least as many columns as rows."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rows is not None and self.cols is not None and self.cols < self.rows:
            raise ValueError("Wide matrices must satisfy cols >= rows.")

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_wide(value, shape=self.shape)


@dataclass(frozen=True)
class ColumnOrthogonal(Tall):
    r"""Domain of tall matrices with orthonormal columns."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_column_orthogonal(value, shape=self.shape)


@dataclass(frozen=True)
class RowOrthogonal(Wide):
    r"""Domain of wide matrices with orthonormal rows."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_row_orthogonal(value, shape=self.shape)


@dataclass(frozen=True)
class RowStochastic(Rectangular):
    r"""Domain of row-stochastic matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_row_stochastic(value, shape=self.shape)


@dataclass(frozen=True)
class ColumnStochastic(Rectangular):
    r"""Domain of column-stochastic matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_column_stochastic(value, shape=self.shape)


@dataclass(frozen=True)
class DoublyStochastic(Square):
    r"""Domain of doubly stochastic square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_doubly_stochastic(value, size=self.size)


@dataclass(frozen=True)
class RowCentered(Rectangular):
    r"""Domain of matrices whose rows sum to zero."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_row_centered(value, shape=self.shape)


@dataclass(frozen=True)
class ColumnCentered(Rectangular):
    r"""Domain of matrices whose columns sum to zero."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_column_centered(value, shape=self.shape)


@dataclass(frozen=True)
class DoublyCentered(Rectangular):
    r"""Domain of matrices whose rows and columns both sum to zero."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_doubly_centered(value, shape=self.shape)


@dataclass(frozen=True)
class Symmetric(Square):
    r"""Domain of symmetric square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_symmetric(value, size=self.size)


@dataclass(frozen=True)
class SkewSymmetric(Square):
    r"""Domain of skew-symmetric square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_skew_symmetric(value, size=self.size)


@dataclass(frozen=True)
class LowRank(Rectangular):
    r"""Domain of rectangular matrices with optional rank bound."""

    _: KW_ONLY
    rank: Final[int | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        if self.rank is None:
            return super().check(value)
        return tests.is_low_rank(value, self.rank, shape=self.shape)

    @overload
    def __call__(self, /, *, rank: int | None = None) -> Self: ...
    @overload
    def __call__(self, rows: int, cols: int, /, *, rank: int | None = None) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        rank: int | None = None,
    ) -> Self:
        return self.__class__(rows, cols, rank=rank)


@dataclass(frozen=True)
class RankOne(LowRank):
    r"""Domain of rank-one rectangular matrices."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rank is None:
            object.__setattr__(self, "rank", 1)
        elif self.rank != 1:
            raise ValueError("Rank must be 1.")

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_rank_one(value, shape=self.shape)


@dataclass(frozen=True)
class LowRankSquare(Square, LowRank):
    r"""Domain of square matrices with optional rank bound."""

    def check(self, value: Tensor, /) -> Tensor:
        if self.rank is None:
            return tests.is_square(value, shape=self.shape)
        return tests.is_low_rank_square(value, self.rank, size=self.size)

    @overload
    def __call__(
        self, size: int | None = None, /, *, rank: int | None = None
    ) -> Self: ...
    @overload
    def __call__(self, rows: int, cols: int, /, *, rank: int | None = None) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        rank: int | None = None,
    ) -> Self:
        return self.__class__(rows, cols, rank=rank)


@dataclass(frozen=True)
class LowRankSymmetric(LowRankSquare):
    r"""Domain of matrices of the form $UVᵀ + VUᵀ$ with optional factor rank bound."""

    def check(self, value: Tensor, /) -> Tensor:
        if self.rank is None:
            return tests.is_symmetric(value, size=self.size)
        return tests.is_low_rank_symmetric(value, self.rank, size=self.size)


@dataclass(frozen=True)
class LowRankSkewSymmetric(LowRankSquare):
    r"""Domain of matrices of the form $UVᵀ - VUᵀ$ with optional factor rank bound."""

    def check(self, value: Tensor, /) -> Tensor:
        if self.rank is None:
            return tests.is_skew_symmetric(value, size=self.size)
        return tests.is_low_rank_skew_symmetric(value, self.rank, size=self.size)


@dataclass(frozen=True)
class LeftInvertible(Tall):
    r"""Domain of left-invertible tall matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_left_invertible(value, shape=self.shape)


@dataclass(frozen=True)
class RightInvertible(Wide):
    r"""Domain of right-invertible wide matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_right_invertible(value, shape=self.shape)


@dataclass(frozen=True)
class Normal(Square):
    r"""Domain of normal square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_normal(value, size=self.size)


@dataclass(frozen=True)
class Orthogonal(Normal):
    r"""Domain of orthogonal square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_orthogonal(value, size=self.size)


@dataclass(frozen=True)
class SpecialOrthogonal(Orthogonal):
    r"""Domain of special orthogonal square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_special_orthogonal(value, size=self.size)


@dataclass(frozen=True)
class Projection(Square):
    r"""Domain of idempotent square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_projection(value, size=self.size)


@dataclass(frozen=True)
class OrthogonalProjection(Projection):
    r"""Domain of symmetric idempotent square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_orthogonal_projection(value, size=self.size)


@dataclass(frozen=True)
class PositiveSemidefinite(Symmetric):
    r"""Domain of symmetric positive semidefinite matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_positive_semidefinite(value, size=self.size)


@dataclass(frozen=True)
class PositiveDefinite(PositiveSemidefinite):
    r"""Domain of symmetric positive definite matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_positive_definite(value, size=self.size)


@dataclass(frozen=True)
class NegativeSemidefinite(Symmetric):
    r"""Domain of symmetric negative semidefinite matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_negative_semidefinite(value, size=self.size)


@dataclass(frozen=True)
class NegativeDefinite(NegativeSemidefinite):
    r"""Domain of symmetric negative definite matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_negative_definite(value, size=self.size)


@dataclass(frozen=True)
class Traceless(Square):
    r"""Domain of traceless square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_traceless(value, size=self.size)


@dataclass(frozen=True)
class Symplectic(EvenSquare):
    r"""Domain of symplectic even square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_symplectic(value, size=self.size)


@dataclass(frozen=True)
class Hamiltonian(EvenSquare):
    r"""Domain of Hamiltonian even square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_hamiltonian(value, size=self.size)


@dataclass(frozen=True)
class Toeplitz(Rectangular):
    r"""Domain of matrices that are constant along diagonals."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_toeplitz(value, shape=self.shape)


@dataclass(frozen=True)
class Banded(Toeplitz):
    r"""Domain of banded matrices."""

    _: KW_ONLY
    lower: Final[int | None] = None
    upper: Final[int | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        if self.lower is None or self.upper is None:
            return super().check(value)
        return tests.is_banded(value, self.lower, self.upper, shape=self.shape)

    @overload
    def __call__(
        self, /, *, lower: int | None = None, upper: int | None = None
    ) -> Self: ...
    @overload
    def __call__(
        self,
        rows: int,
        cols: int,
        /,
        *,
        lower: int | None = None,
        upper: int | None = None,
    ) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        lower: int | None = None,
        upper: int | None = None,
    ) -> Self:
        return self.__class__(rows, cols, lower=lower, upper=upper)


@dataclass(frozen=True)
class Circulant(Square, Toeplitz):
    r"""Domain of circulant square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_circulant(value, size=self.size)


@dataclass(frozen=True)
class Tridiagonal(Circulant):
    r"""Domain of tridiagonal square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_tridiagonal(value, size=self.size)


@dataclass(frozen=True)
class Diagonal(Tridiagonal):
    r"""Domain of diagonal square matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_diagonal(value, size=self.size)


@dataclass(frozen=True)
class NegativeDiagonal(Diagonal, NegativeDefinite):
    r"""Domain of diagonal matrices with strictly negative diagonal entries."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_negative_diagonal(value, size=self.size)


@dataclass(frozen=True)
class PositiveDiagonal(Diagonal, PositiveDefinite):
    r"""Domain of diagonal matrices with strictly positive diagonal entries."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_positive_diagonal(value, size=self.size)


@dataclass(frozen=True)
class PositiveScalarMatrix(PositiveDiagonal):
    r"""Domain of matrices of the form $σI$ with $σ > 0$."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_positive_scalar_matrix(value, size=self.size)


@dataclass(frozen=True)
class BlockDiagonal(Square):
    r"""Domain of square matrices supported on diagonal blocks."""

    _: KW_ONLY
    block_sizes: Final[tuple[int, ...] | None] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.block_sizes is None:
            return
        if not self.block_sizes or any(
            block_size <= 0 for block_size in self.block_sizes
        ):
            raise ValueError(
                "block_sizes must be a non-empty tuple of positive integers."
            )
        if self.size is not None and sum(self.block_sizes) != self.size:
            raise ValueError("Sum of block_sizes must equal the matrix size.")

    def check(self, value: Tensor, /) -> Tensor:
        if self.block_sizes is None:
            return tests.is_square(value, shape=self.shape)
        return tests.is_block_diagonal(
            value, self.block_sizes, size=self.size if self.size is not None else None
        )

    @overload
    def __call__(
        self, size: int | None = None, /, *, block_sizes: tuple[int, ...] | None = None
    ) -> Self: ...
    @overload
    def __call__(
        self, rows: int, cols: int, /, *, block_sizes: tuple[int, ...] | None = None
    ) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        block_sizes: tuple[int, ...] | None = None,
    ) -> Self:
        return self.__class__(rows, cols, block_sizes=block_sizes)


@dataclass(frozen=True)
class Triangular(Rectangular):
    r"""Domain of lower or upper triangular matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_triangular(value, shape=self.shape)


@dataclass(frozen=True)
class LowerTriangular(Triangular):
    r"""Domain of lower triangular matrices."""

    _: KW_ONLY
    lower: Final[int | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        if self.lower is None:
            return tests.is_lower_triangular(value, shape=self.shape)
        return tests.is_lower_triangular(value, self.lower, shape=self.shape)

    @overload
    def __call__(self, /, *, lower: int | None = None) -> Self: ...
    @overload
    def __call__(
        self, rows: int, cols: int, /, *, lower: int | None = None
    ) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        lower: int | None = None,
    ) -> Self:
        return self.__class__(rows, cols, lower=lower)


@dataclass(frozen=True)
class UpperTriangular(Triangular):
    r"""Domain of upper triangular matrices."""

    _: KW_ONLY
    upper: Final[int | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        if self.upper is None:
            return tests.is_upper_triangular(value, shape=self.shape)
        return tests.is_upper_triangular(value, self.upper, shape=self.shape)

    @overload
    def __call__(self, /, *, upper: int | None = None) -> Self: ...
    @overload
    def __call__(
        self, rows: int, cols: int, /, *, upper: int | None = None
    ) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        upper: int | None = None,
    ) -> Self:
        return self.__class__(rows, cols, upper=upper)


@dataclass(frozen=True)
class Masked(Rectangular):
    r"""Domain of matrices supported on a fixed mask."""

    _: KW_ONLY
    mask: Final[Tensor | None] = None

    def check(self, value: Tensor, /) -> Tensor:
        if self.mask is None:
            return super().check(value)
        return tests.is_masked(value, self.mask, shape=self.shape)

    @overload
    def __call__(self, /, *, mask: Tensor | None = None) -> Self: ...
    @overload
    def __call__(
        self, rows: int, cols: int, /, *, mask: Tensor | None = None
    ) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        mask: Tensor | None = None,
    ) -> Self:
        return self.__class__(rows, cols, mask=mask)


@dataclass(frozen=True)
class LipschitzBounded(Rectangular):
    r"""Domain of matrices with bounded Lipschitz constant."""

    _: KW_ONLY
    lipschitz_bound: Final[float | None] = None

    def __post_init__(self):
        if self.lipschitz_bound is None:
            pass
        elif self.lipschitz_bound < 0:
            raise ValueError("lipschitz_bound must be >= 0")

    def check(self, value: Tensor, /) -> Tensor:
        if self.lipschitz_bound is None:
            return super().check(value)
        return tests.is_lipschitz_bounded(value, self.lipschitz_bound, shape=self.shape)

    @overload
    def __call__(self, /, *, lipschitz_bound: float | None = None) -> Self: ...
    @overload
    def __call__(
        self, rows: int, cols: int, /, *, lipschitz_bound: float | None = None
    ) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        lipschitz_bound: float | None = None,
    ) -> Self:
        return self.__class__(rows, cols, lipschitz_bound=lipschitz_bound)


@dataclass(frozen=True)
class SpectralNormalized(LipschitzBounded):
    r"""Domain of matrices with unit spectral norm."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lipschitz_bound is None:
            object.__setattr__(self, "lipschitz_bound", 1.0)
        elif self.lipschitz_bound != 1.0:
            raise ValueError(
                "Spectral normalized matrices must have lipschitz_bound=1.0."
            )

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_spectral_normalized(value, shape=self.shape)


@dataclass(frozen=True)
class Contraction(LipschitzBounded):
    r"""Domain of contraction matrices."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lipschitz_bound is not None and not (0 < self.lipschitz_bound <= 1):
            raise ValueError("Contraction bound must satisfy 0 < lipschitz_bound <= 1.")

    def check(self, value: Tensor, /) -> Tensor:
        if self.lipschitz_bound is None:
            return super().check(value)
        return tests.is_contraction(value, self.lipschitz_bound, shape=self.shape)

    @overload
    def __call__(self, /, *, lipschitz_bound: float | None = 1.0) -> Self: ...
    @overload
    def __call__(
        self, rows: int, cols: int, /, *, lipschitz_bound: float | None = 1.0
    ) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        lipschitz_bound: float | None = 1.0,
    ) -> Self:
        return self.__class__(rows, cols, lipschitz_bound=lipschitz_bound)


@dataclass(frozen=True)
class DiagonallyDominant(Square):
    r"""Domain of diagonally dominant square matrices."""

    _: KW_ONLY
    strict: Final[bool] = False

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_diagonally_dominant(value, size=self.size, strict=self.strict)

    @overload
    def __call__(self, size: int | None = None, /, *, strict: bool = False) -> Self: ...
    @overload
    def __call__(self, rows: int, cols: int, /, *, strict: bool = False) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        strict: bool = False,
    ) -> Self:
        return self.__class__(rows, cols, strict=strict)


@dataclass(frozen=True)
class ForwardStable(Rectangular):
    r"""Domain of forward stable matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_forward_stable(value)


@dataclass(frozen=True)
class BackwardStable(Rectangular):
    r"""Domain of backward stable matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_backward_stable(value)


@dataclass(frozen=True)
class Boolean(Rectangular):
    r"""Domain of matrices whose entries are only zeros and ones."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_boolean(value, shape=self.shape)


@dataclass(frozen=True)
class Zero(Boolean):
    r"""Domain of matrices whose entries are only zeros."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_zero(value, shape=self.shape)


@dataclass(frozen=True)
class Ones(Boolean):
    r"""Domain of matrices whose entries are only ones."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_ones(value, shape=self.shape)


@dataclass(frozen=True)
class OneHot(Boolean):
    r"""Domain of matrices with exactly one 1 entry and zeros elsewhere."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_one_hot(value, shape=self.shape)


@dataclass(frozen=True)
class Sparse(Rectangular):
    r"""Domain of matrices with sufficiently many exact zero entries."""

    _: KW_ONLY
    sparsity: Final[float | None] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sparsity is not None and not 0.0 <= self.sparsity <= 1.0:
            raise ValueError("Expected sparsity in [0, 1].")

    def check(self, value: Tensor, /) -> Tensor:
        return Rectangular.check(self, value) & tests.is_sparse(value, self.sparsity)

    @overload
    def __call__(self, /, *, sparsity: float | None = None) -> Self: ...
    @overload
    def __call__(
        self, rows: int, cols: int, /, *, sparsity: float | None = None
    ) -> Self: ...
    def __call__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        /,
        *,
        sparsity: float | None = None,
    ) -> Self:
        return self.__class__(rows, cols, sparsity=sparsity)


@dataclass(frozen=True)
class Permutation(DoublyStochastic):
    r"""Domain of permutation matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_permutation(value, size=self.size)


@dataclass(frozen=True)
class Identity(Diagonal, Permutation):
    r"""Domain of identity matrices."""

    def check(self, value: Tensor, /) -> Tensor:
        return tests.is_identity(value, size=self.size)


class MatrixDomains(MatrixDomain, PosetEnum):
    r"""Enumeration of some matrix domains."""

    ALIASES: ClassVar[Mapping[str, Self]]

    ANY = Rectangular()  # top node
    NONE = Empty()  # bottom node

    REAL = Placeholder(name="real")
    COMPLEX = Placeholder(name="complex")
    DISCRETE = Placeholder(name="discrete")
    BOOLEAN = Boolean()  # a matrix of only zeros and ones
    ZERO = Zero()  # a matrix of only zeros
    ONES = Ones()  # a matrix of only ones
    IDENTITY = Identity()  # the identity matrix
    EYE = Identity()  # alias

    ONE_HOT = OneHot()  # a single 1 entry and 0 elsewhere

    RECTANGULAR = Rectangular()  # m × n matrices
    TALL = Tall()  # m × n matrices with m ≥ n
    WIDE = Wide()  # m × n matrices with m ≤ n
    COLUMN_ORTHOGONAL = ColumnOrthogonal()  # m × n matrices with QᵀQ = 𝕀ₙ
    ROW_ORTHOGONAL = RowOrthogonal()  # m × n matrices with QQᵀ = 𝕀ₘ
    SQUARE = Square()  # n × n matrices

    # rank
    LOW_RANK = LowRank()  # UVᵀ
    LOW_RANK_SQUARE = LowRankSquare()
    LOW_RANK_SYMMETRIC = LowRankSymmetric()  # UVᵀ + VUᵀ
    LOW_RANK_SKEW_SYMMETRIC = LowRankSkewSymmetric()  # UVᵀ - VUᵀ
    RANK_ONE = RankOne()  # uvᵀ

    # determinant-based
    SINGULAR = Placeholder(name="singular")  # det=0
    LEFT_INVERTIBLE = LeftInvertible()  # full column rank, admits L with LA = 𝕀
    RIGHT_INVERTIBLE = RightInvertible()  # full row rank, admits R with AR = 𝕀
    INVERTIBLE = Placeholder(name="invertible")  # GLₙ(R) (det≠0)
    LOWER_INVERTIBLE = Placeholder(name="lower-invertible")
    UPPER_INVERTIBLE = Placeholder(name="upper-invertible")
    CHOLESKY_FACTOR = Placeholder(name="cholesky-factor")
    GENERAL_LINEAR = Placeholder(name="invertible")  # alias
    SPECIAL_LINEAR = Placeholder(name="unit-determinant")  # alias
    UNIT_DETERMINANT = Placeholder(name="unit-determinant")  # SLₙ(R) (det=1)
    POSITIVE_DETERMINANT = Placeholder(name="positive-determinant")  # GLₙ⁺(R) (det>0)
    NEGATIVE_DETERMINANT = Placeholder(name="negative-determinant")  # GLₙ⁻(R) (det<0)
    UNIMODULAR = Placeholder(name="modular")  # discrete and det A = ±1

    # symmetry / entry based
    SYMMETRIC = Symmetric()  # 𝕊ₙ(R)
    SKEW_SYMMETRIC = SkewSymmetric()

    TOEPLITZ = Toeplitz()  # constant along diagonals
    BANDED = Banded()
    CIRCULANT = Circulant()  # constant along diagonals, wrap around
    TRIDIAGONAL = Tridiagonal()
    DIAGONAL = Diagonal()
    NEGATIVE_DIAGONAL = NegativeDiagonal()
    POSITIVE_DIAGONAL = PositiveDiagonal()
    POSITIVE_SCALAR_MATRIX = PositiveScalarMatrix()

    # diagonal conditions
    POSITIVE_DIAGONAL_ENTRIES = Placeholder(
        name="positive-diagonal-entries"
    )  # Aᵢᵢ > 0 for all i
    NEGATIVE_DIAGONAL_ENTRIES = Placeholder(
        name="negative-diagonal-entries"
    )  # Aᵢᵢ < 0 for all i
    ZERO_DIAGONAL = Placeholder(name="zero-diagonal")  # Aᵢᵢ = 0 for all i
    UNIDIAGONAL = Placeholder(name="unidagonal")  # Aᵢᵢ = 1 for all i
    TRACELESS = Traceless()  # ∑ᵢ Aᵢᵢ = 0

    HANKEL = Placeholder(name="hankel")  # constant along anti-diagonals

    # eigenvalues
    POSITIVE_DEFINITE = PositiveDefinite()  # 𝕊ₙ⁺(ℝ)
    NEGATIVE_DEFINITE = NegativeDefinite()  # 𝕊ₙ⁻(ℝ)
    POSITIVE_SEMIDEFINITE = PositiveSemidefinite()  # 𝕊ₙ⁺(ℝ) ∪ {0}
    NEGATIVE_SEMIDEFINITE = NegativeSemidefinite()  # 𝕊ₙ⁻(ℝ) ∪ {0}

    CONTRACTION = Contraction()  # ‖A‖₂ < 1
    SPECTRAL_NORMALIZED = SpectralNormalized()  # ‖A‖₂ = 1
    LIPSCHITZ_BOUNDED = LipschitzBounded()  # ‖A‖₂ ≤ C
    DIAGONALLY_DOMINANT = DiagonallyDominant()  # |Aᵢᵢ| ≥ ∑_{j≠i} |Aᵢⱼ| for all i

    NORMAL = Normal()  # AᵀA = AAᵀ
    ORTHOGONAL = Orthogonal()  # Oₙ(R)
    CAYLEY_ORTHOGONAL = Placeholder(
        name="cayley-orthogonal"
    )  # {Q ∈ SOₙ(n) ∣ -1 ∉ spec(Q)}
    SPECIAL_ORTHOGONAL = SpecialOrthogonal()  # SOₙ(R)

    EVEN_SQUARE = EvenSquare()  # 2n × 2n matrices
    SYMPLECTIC = Symplectic()  # 2n×2n with AᵀJA = J for J=[0, I;-I, 0]
    HAMILTONIAN = Hamiltonian()  # 2n×2n with (JA)ᵀ = JA for J=[0, I;-I, 0]

    TRIANGULAR = Triangular()  # lower or upper
    UPPER_TRIANGULAR = UpperTriangular()
    LOWER_TRIANGULAR = LowerTriangular()

    ROW_CENTERED = RowCentered()  # A𝟏 = 𝟎
    COLUMN_CENTERED = ColumnCentered()  # Aᵀ𝟏 = 𝟎
    DOUBLY_CENTERED = DoublyCentered()  # A𝟏 = 𝟎 and Aᵀ𝟏 = 𝟎
    CENTERING = Placeholder(name="centering")  # 𝕀ₙ - 1/n𝟏ₙ𝟏ₙᵀ special centering matrix
    INTENSITY = Placeholder(
        name="intensity"
    )  # Aᵢᵢ = -∑_{j≠i} Aᵢⱼ for all i, Aᵢⱼ ≥ 0 for i≠j
    # ⇝ row-centered, nonpositive diagonal, diagonally dominant

    ROW_STOCHASTIC = RowStochastic()  # nonnegative, A𝟏 = 𝟏
    COLUMN_STOCHASTIC = ColumnStochastic()  # nonnegative, Aᵀ𝟏 = 𝟏
    DOUBLY_STOCHASTIC = DoublyStochastic()
    PERMUTATION = Permutation()

    IDEMPOTENT = Placeholder(name="idempotent")  # Pᵏ = P for some k≥2
    PROJECTION = Projection()  # P² = P  (P=QQ⁺ + QZ(I-QQ⁺) for some Q,Z)
    ORTHOGONAL_PROJECTION = OrthogonalProjection()  # P² = P, P symmetric (P=QQ⁺)
    NILPOTENT = Placeholder(name="nilpotent")  # Aᵏ = 0 for some k≥2

    # special matrices
    STANDARD_NILPOTENT = Placeholder(
        name="canonical-nilpotent"
    )  # standard nilpotent matrix
    STANDARD_SYMPLECTIC = Placeholder(name="standard-symplectic")  # [0, 𝕀; -𝕀, 0]
    HOUSEHOLDER = Placeholder(name="householder")
    GIVENS_ROTATION = Placeholder(name="givens-rotation")
    HADAMARD = Placeholder(name="hadamard")  # entries ±1, HHᵀ=n𝕀
    JORDAN_BLOCK = Placeholder(name="jordan-block")  # λI + N, N is standard nilpotent

    BLOCK_DIAGONAL = BlockDiagonal()
    JORDAN = Placeholder(name="jordan")  # block diagonal with Jordan blocks

    # TODO: graph theory (degree, adjacency, incidence, Laplacian)

    # tag-like
    MASKED = Masked()  # X⊙M = X for some mask M
    SPARSE = Sparse()  # many 0 entries
    EFFICIENTLY_INVERTIBLE = Placeholder(name="efficiently-invertible")
    FORWARD_STABLE = ForwardStable()
    BACKWARD_STABLE = BackwardStable()
    DIAGONALIZABLE = Placeholder(name="diagonalizable")

    @property
    def rows(self) -> int | None:
        return self.value.rows

    @property
    def cols(self) -> int | None:
        return self.value.cols

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


M = MatrixDomains  # temporary alias
# pyrefly: ignore[bad-assignment]
MatrixDomains.KNOWN_MEETS = (  # pyright: ignore[reportAttributeAccessIssue]
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
    (M.IDENTITY, M.POSITIVE_DEFINITE & M.ORTHOGONAL),
    (M.IDENTITY, M.DIAGONAL & M.PERMUTATION),
    # other
    (M.CHOLESKY_FACTOR, M.LOWER_INVERTIBLE & M.POSITIVE_DIAGONAL_ENTRIES),
    (M.DIAGONAL, M.LOWER_TRIANGULAR & M.UPPER_TRIANGULAR),
    (M.DOUBLY_CENTERED, M.ROW_CENTERED & M.COLUMN_CENTERED),
    (M.DOUBLY_STOCHASTIC, M.SQUARE & M.ROW_STOCHASTIC & M.COLUMN_STOCHASTIC),
    (M.INVERTIBLE, M.LEFT_INVERTIBLE & M.RIGHT_INVERTIBLE),
    (M.LOWER_INVERTIBLE, M.LOWER_TRIANGULAR & M.INVERTIBLE),
    (M.LOW_RANK_SKEW_SYMMETRIC, M.LOW_RANK_SQUARE & M.SKEW_SYMMETRIC),
    (M.LOW_RANK_SQUARE, M.LOW_RANK & M.SQUARE),
    (M.LOW_RANK_SYMMETRIC, M.LOW_RANK_SQUARE & M.SYMMETRIC),
    (M.NEGATIVE_DEFINITE, M.NEGATIVE_SEMIDEFINITE & M.INVERTIBLE),
    (M.NEGATIVE_DIAGONAL, M.DIAGONAL & M.NEGATIVE_DEFINITE),
    (M.ORTHOGONAL, M.COLUMN_ORTHOGONAL & M.ROW_ORTHOGONAL),
    (M.ORTHOGONAL_PROJECTION, M.PROJECTION & M.SYMMETRIC),
    (M.PERMUTATION, M.ORTHOGONAL & M.DOUBLY_STOCHASTIC),
    (M.POSITIVE_DEFINITE, M.POSITIVE_SEMIDEFINITE & M.INVERTIBLE),
    (M.POSITIVE_DIAGONAL, M.DIAGONAL & M.POSITIVE_DEFINITE),
    (M.SPECIAL_ORTHOGONAL, M.ORTHOGONAL & M.UNIT_DETERMINANT),
    (M.SQUARE, M.ROW_STOCHASTIC & M.COLUMN_STOCHASTIC),  # theorem
    (M.SQUARE, M.TALL & M.WIDE),
    (M.UPPER_INVERTIBLE, M.UPPER_TRIANGULAR & M.INVERTIBLE),
)
# pyrefly: ignore[bad-assignment]
MatrixDomains.KNOWN_SUPERTYPES = MappingProxyType({  # pyright: ignore[reportAttributeAccessIssue]
    M.BANDED: {M.TOEPLITZ},
    M.BOOLEAN: {M.RECTANGULAR},
    M.CAYLEY_ORTHOGONAL: {M.SPECIAL_ORTHOGONAL},
    M.CHOLESKY_FACTOR: {M.LOWER_INVERTIBLE, M.POSITIVE_DIAGONAL_ENTRIES},
    M.CIRCULANT: {M.TOEPLITZ, M.SQUARE},
    M.COLUMN_ORTHOGONAL: {M.LEFT_INVERTIBLE, M.SPECTRAL_NORMALIZED},
    M.COLUMN_STOCHASTIC: {M.RECTANGULAR},
    M.CONTRACTION: {M.LIPSCHITZ_BOUNDED},
    M.DIAGONAL: {M.SYMMETRIC, M.TRIDIAGONAL},
    M.DIAGONALLY_DOMINANT: {M.SQUARE},
    M.DOUBLY_CENTERED: {M.ROW_CENTERED, M.COLUMN_CENTERED},
    M.DOUBLY_STOCHASTIC: {M.SQUARE & M.ROW_STOCHASTIC, M.COLUMN_STOCHASTIC},
    M.EVEN_SQUARE: {M.SQUARE},
    M.HAMILTONIAN: {M.EVEN_SQUARE, M.TRACELESS},
    M.HANKEL: {M.RECTANGULAR},
    M.IDENTITY: {M.POSITIVE_SCALAR_MATRIX},
    M.INVERTIBLE: {M.LEFT_INVERTIBLE, M.RIGHT_INVERTIBLE},
    M.LEFT_INVERTIBLE: {M.TALL},
    M.LIPSCHITZ_BOUNDED: {M.RECTANGULAR},
    M.LOWER_INVERTIBLE: {M.LOWER_TRIANGULAR, M.INVERTIBLE},
    M.LOWER_TRIANGULAR: {M.TRIANGULAR},
    M.LOW_RANK: {M.RECTANGULAR},
    M.LOW_RANK_SKEW_SYMMETRIC: {M.LOW_RANK_SQUARE, M.SKEW_SYMMETRIC},
    M.LOW_RANK_SQUARE: {M.LOW_RANK, M.SQUARE},
    M.LOW_RANK_SYMMETRIC: {M.LOW_RANK_SQUARE, M.SYMMETRIC},
    M.NEGATIVE_DEFINITE: {M.NEGATIVE_SEMIDEFINITE, M.INVERTIBLE},
    M.NEGATIVE_DETERMINANT: {M.INVERTIBLE},
    M.NEGATIVE_DIAGONAL: {M.NEGATIVE_DIAGONAL_ENTRIES},
    M.NEGATIVE_DIAGONAL_ENTRIES: {M.RECTANGULAR},
    M.NEGATIVE_SEMIDEFINITE: {M.SYMMETRIC},
    M.NORMAL: {M.SQUARE},
    M.ONES: {M.BOOLEAN},
    M.ONE_HOT: {M.BOOLEAN},
    M.ORTHOGONAL: {M.COLUMN_ORTHOGONAL, M.ROW_ORTHOGONAL},
    M.ORTHOGONAL_PROJECTION: {M.PROJECTION, M.SYMMETRIC},
    M.PERMUTATION: {M.ORTHOGONAL, M.DOUBLY_STOCHASTIC},
    M.POSITIVE_DEFINITE: {M.POSITIVE_SEMIDEFINITE, M.INVERTIBLE},
    M.POSITIVE_DETERMINANT: {M.INVERTIBLE},
    M.POSITIVE_DIAGONAL: {M.POSITIVE_DIAGONAL_ENTRIES},
    M.POSITIVE_DIAGONAL_ENTRIES: {M.RECTANGULAR},
    M.POSITIVE_SCALAR_MATRIX: {M.POSITIVE_DIAGONAL},
    M.POSITIVE_SEMIDEFINITE: {M.SYMMETRIC},
    M.PROJECTION: {M.SQUARE},
    M.RANK_ONE: {M.LOW_RANK},
    M.RIGHT_INVERTIBLE: {M.WIDE},
    M.ROW_ORTHOGONAL: {M.RIGHT_INVERTIBLE, M.SPECTRAL_NORMALIZED},
    M.ROW_STOCHASTIC: {M.RECTANGULAR},
    M.SINGULAR: {M.SQUARE},
    M.SKEW_SYMMETRIC: {M.SQUARE, M.NORMAL, M.ZERO_DIAGONAL},
    M.SPARSE: {M.BOOLEAN},
    M.SPECIAL_ORTHOGONAL: {M.ORTHOGONAL, M.UNIT_DETERMINANT},
    M.SPECTRAL_NORMALIZED: {M.RECTANGULAR, M.LIPSCHITZ_BOUNDED},
    M.SQUARE: {M.TALL, M.WIDE},
    M.STANDARD_SYMPLECTIC: {M.SYMPLECTIC, M.SKEW_SYMMETRIC},
    M.SYMMETRIC: {M.SQUARE, M.NORMAL},
    M.SYMPLECTIC: {M.EVEN_SQUARE, M.UNIT_DETERMINANT},
    M.TALL: {M.RECTANGULAR},
    M.TOEPLITZ: {M.RECTANGULAR},
    M.TRACELESS: {M.SQUARE},
    M.TRIDIAGONAL: {M.CIRCULANT},
    M.UNIT_DETERMINANT: {M.POSITIVE_DETERMINANT},
    M.UPPER_INVERTIBLE: {M.UPPER_TRIANGULAR, M.INVERTIBLE},
    M.UPPER_TRIANGULAR: {M.TRIANGULAR},
    M.WIDE: {M.RECTANGULAR},
    M.ZERO: {M.BOOLEAN},
    M.ZERO_DIAGONAL: {M.TRACELESS},
})  # fmt: skip
MatrixDomains.KNOWN_SUBTYPES = MappingProxyType({
    # These relationships are not modeled by inheritance.
    M.SPARSE: {
        M.ZERO, M.PERMUTATION, M.ONE_HOT, M.BANDED,
        M.BLOCK_DIAGONAL, M.JORDAN_BLOCK, M.STANDARD_NILPOTENT,
        M.STANDARD_SYMPLECTIC,
    },
    M.EFFICIENTLY_INVERTIBLE: {
        M.SPARSE, M.PERMUTATION, M.ORTHOGONAL,
        M.TRIANGULAR, M.DIAGONAL, M.TRIDIAGONAL,
    },
})  # fmt: skip
MatrixDomains.ALIASES = MappingProxyType({
    "any"                        : M.ANY,
    "backward-stable"            : M.BACKWARD_STABLE,
    "banded"                     : M.BANDED,
    "block-diagonal"             : M.BLOCK_DIAGONAL,
    "boolean"                    : M.BOOLEAN,
    "canonical-nilpotent"        : M.STANDARD_NILPOTENT,
    "cayley-orthogonal"          : M.CAYLEY_ORTHOGONAL,
    "centering"                  : M.CENTERING,
    "cholesky-factor"            : M.CHOLESKY_FACTOR,
    "circulant"                  : M.CIRCULANT,
    "column-centered"            : M.COLUMN_CENTERED,
    "column-orthogonal"          : M.COLUMN_ORTHOGONAL,
    "column-stochastic"          : M.COLUMN_STOCHASTIC,
    "complex"                    : M.COMPLEX,
    "contraction"                : M.CONTRACTION,
    "diagonal"                   : M.DIAGONAL,
    "diagonalizable"             : M.DIAGONALIZABLE,
    "diagonally-dominant"        : M.DIAGONALLY_DOMINANT,
    "discrete"                   : M.DISCRETE,
    "doubly-centered"            : M.DOUBLY_CENTERED,
    "doubly-stochastic"          : M.DOUBLY_STOCHASTIC,
    "efficiently-invertible"     : M.EFFICIENTLY_INVERTIBLE,
    "empty"                      : M.NONE,
    "even-square"                : M.EVEN_SQUARE,
    "eye"                        : M.EYE,
    "forward-stable"             : M.FORWARD_STABLE,
    "general-linear"             : M.GENERAL_LINEAR,
    "givens-rotation"            : M.GIVENS_ROTATION,
    "hadamard"                   : M.HADAMARD,
    "hamiltonian"                : M.HAMILTONIAN,
    "hankel"                     : M.HANKEL,
    "householder"                : M.HOUSEHOLDER,
    "idempotent"                 : M.IDEMPOTENT,
    "identity"                   : M.IDENTITY,
    "intensity"                  : M.INTENSITY,
    "invertible"                 : M.INVERTIBLE,
    "jordan"                     : M.JORDAN,
    "jordan-block"               : M.JORDAN_BLOCK,
    "left-invertible"            : M.LEFT_INVERTIBLE,
    "lipschitz-bounded"          : M.LIPSCHITZ_BOUNDED,
    "low-rank"                   : M.LOW_RANK,
    "low-rank-skew-symmetric"    : M.LOW_RANK_SKEW_SYMMETRIC,
    "low-rank-square"            : M.LOW_RANK_SQUARE,
    "low-rank-symmetric"         : M.LOW_RANK_SYMMETRIC,
    "lower-invertible"           : M.LOWER_INVERTIBLE,
    "lower-triangular"           : M.LOWER_TRIANGULAR,
    "masked"                     : M.MASKED,
    "negative-definite"          : M.NEGATIVE_DEFINITE,
    "negative-determinant"       : M.NEGATIVE_DETERMINANT,
    "negative-diagonal"          : M.NEGATIVE_DIAGONAL,
    "negative-diagonal-entries"  : M.NEGATIVE_DIAGONAL_ENTRIES,
    "negative-semidefinite"      : M.NEGATIVE_SEMIDEFINITE,
    "nilpotent"                  : M.NILPOTENT,
    "none"                       : M.NONE,
    "normal"                     : M.NORMAL,
    "one-hot"                    : M.ONE_HOT,
    "ones"                       : M.ONES,
    "orthogonal"                 : M.ORTHOGONAL,
    "orthogonal-projection"      : M.ORTHOGONAL_PROJECTION,
    "permutation"                : M.PERMUTATION,
    "positive-definite"          : M.POSITIVE_DEFINITE,
    "positive-determinant"       : M.POSITIVE_DETERMINANT,
    "positive-diagonal"          : M.POSITIVE_DIAGONAL,
    "positive-diagonal-entries"  : M.POSITIVE_DIAGONAL_ENTRIES,
    "positive-scalar-matrix"     : M.POSITIVE_SCALAR_MATRIX,
    "positive-semidefinite"      : M.POSITIVE_SEMIDEFINITE,
    "projection"                 : M.PROJECTION,
    "rank-one"                   : M.RANK_ONE,
    "real"                       : M.REAL,
    "rectangular"                : M.RECTANGULAR,
    "right-invertible"           : M.RIGHT_INVERTIBLE,
    "row-centered"               : M.ROW_CENTERED,
    "row-orthogonal"             : M.ROW_ORTHOGONAL,
    "row-stochastic"             : M.ROW_STOCHASTIC,
    "singular"                   : M.SINGULAR,
    "skew-symmetric"             : M.SKEW_SYMMETRIC,
    "sparse"                     : M.SPARSE,
    "special-linear"             : M.SPECIAL_LINEAR,
    "special-orthogonal"         : M.SPECIAL_ORTHOGONAL,
    "spectral-normalized"        : M.SPECTRAL_NORMALIZED,
    "square"                     : M.SQUARE,
    "standard-symplectic"        : M.STANDARD_SYMPLECTIC,
    "symmetric"                  : M.SYMMETRIC,
    "symplectic"                 : M.SYMPLECTIC,
    "tall"                       : M.TALL,
    "toeplitz"                   : M.TOEPLITZ,
    "traceless"                  : M.TRACELESS,
    "triangular"                 : M.TRIANGULAR,
    "tridiagonal"                : M.TRIDIAGONAL,
    "unidiagonal"                : M.UNIDIAGONAL,
    "unimodular"                 : M.UNIMODULAR,
    "unit-determinant"           : M.UNIT_DETERMINANT,
    "upper-invertible"           : M.UPPER_INVERTIBLE,
    "upper-triangular"           : M.UPPER_TRIANGULAR,
    "wide"                       : M.WIDE,
    "zero"                       : M.ZERO,
    "zero-diagonal"              : M.ZERO_DIAGONAL,
})  # fmt: skip
del M  # remove alias
