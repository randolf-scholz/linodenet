r"""Checks for testing certain matrix properties (type (1,1)-tensors)."""

__all__ = [
    # ABCs & Protocols
    "MatrixTest",
    "MatrixTestWithArgs",
    # is_* checks
    "is_backward_stable",
    "is_banded",
    "is_block_diagonal",
    "is_boolean",
    "is_contraction",
    "is_column_stochastic",
    "is_column_centered",
    "is_column_orthogonal",
    "is_spectral_normalized",
    "is_lipschitz_bounded",
    "is_left_invertible",
    "is_diagonal",
    "is_diagonally_dominant",
    "is_forward_stable",
    "is_hamiltonian",
    "is_identity",
    "is_one_hot",
    "is_low_rank",
    "is_low_rank_square",
    "is_low_rank_skew_symmetric",
    "is_low_rank_symmetric",
    "is_lower_triangular",
    "is_masked",
    "is_negative_diagonal",
    "is_negative_definite",
    "is_negative_semidefinite",
    "is_normal",
    "is_orthogonal",
    "is_ones",
    "is_permutation",
    "is_positive_diagonal",
    "is_positive_definite",
    "is_positive_scalar_matrix",
    "is_positive_semidefinite",
    "is_projection",
    "is_right_invertible",
    "is_doubly_stochastic",
    "is_doubly_centered",
    "is_orthogonal_projection",
    "is_row_orthogonal",
    "is_row_centered",
    "is_row_stochastic",
    "is_special_orthogonal",
    "is_rank_one",
    "is_skew_symmetric",
    "is_square",
    "is_symmetric",
    "is_symplectic",
    "is_traceless",
    "is_toeplitz",
    "is_triangular",
    "is_tridiagonal",
    "is_upper_triangular",
    "is_tall",
    "is_wide",
    "is_zero",
    "is_circulant",
    "is_cholesky_factor",
]

from collections.abc import Callable
from typing import Concatenate, Protocol

import torch
from torch import Tensor

from linodenet.constants import ATOL, RTOL
from signatures import signature


def _matrix_batch_shape(x: Tensor, dim: tuple[int, int], /) -> tuple[int, ...]:
    r"""Return the batch shape after removing the matrix dimensions."""
    m, n = (axis % x.ndim for axis in dim)
    return tuple(size for axis, size in enumerate(x.shape) if axis not in {m, n})


def _full_false(x: Tensor, dim: tuple[int, int], /) -> Tensor:
    r"""Return a boolean tensor filled with `False` over the batch shape."""
    return torch.full(
        _matrix_batch_shape(x, dim), False, dtype=torch.bool, device=x.device
    )


def _has_shape(x: Tensor, shape: tuple[int, int], dim: tuple[int, int], /) -> bool:
    r"""Return whether the selected matrix dimensions match `shape`."""
    m, n = dim
    return x.shape[m] == shape[0] and x.shape[n] == shape[1]


def _has_size(x: Tensor, size: int, dim: tuple[int, int], /) -> bool:
    r"""Return whether the selected matrix dimensions equal `(size, size)`."""
    return _has_shape(x, (size, size), dim)


class MatrixTest(Protocol):
    r"""Protocol for testing certain matrix property."""

    @signature("(..., m, n) -> bool[(...)]")
    def __call__(
        self,
        x: Tensor,
        /,
        *,
        dim: tuple[int, int] = (-2, -1),
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> Tensor:
        r"""Check whether the given matrix belongs to a matrix group/manifold.

        Note:
            - There are different kinds of matrix groups, which are not cleanly separated
              here at the moment (additive, multiplicative, Lie groups, ...).
            - JIT does not support positional-only and keyword-only arguments.
              So they are only used in the protocol.
        """
        ...


type MatrixTestWithArgs = Callable[Concatenate[Tensor, ...], Tensor]


# region is_* checks -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
@signature("(..., m, n) -> bool[(...)]")
def is_low_rank(
    x: Tensor,
    /,
    rank: int,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is low-rank."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    # move target dims to -1 and -2
    x = x.movedim(dim, (-2, -1))
    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return ranks <= rank


@signature("(..., m, n) -> bool[(...)]")
def is_rank_one(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is rank-1."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    # move target dims to -1 and -2
    x = x.movedim(dim, (-2, -1))
    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return ranks <= 1


@signature("(..., n, n) -> bool[(...)]")
def is_low_rank_square(
    x: Tensor,
    /,
    rank: int,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is square with rank at most `rank`."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return ranks <= rank


@signature("(..., n, n) -> bool[(...)]")
def is_low_rank_symmetric(
    x: Tensor,
    /,
    rank: int,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symmetric with rank at most `2⋅rank`."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    symmetric = is_symmetric(x, dim=(-2, -1), rtol=rtol, atol=atol)
    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return symmetric & (ranks <= 2 * rank)


@signature("(..., n, n) -> bool[(...)]")
def is_low_rank_skew_symmetric(
    x: Tensor,
    /,
    rank: int,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is skew-symmetric with rank at most `2⋅rank`."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    skew_symmetric = is_skew_symmetric(x, dim=(-2, -1), rtol=rtol, atol=atol)
    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return skew_symmetric & (ranks <= 2 * rank)


@signature("(..., m, n) -> bool[()]")
def is_square(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor is square along the given dimensions."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    return torch.tensor(
        x.shape[dim[0]] == x.shape[dim[1]],
        dtype=torch.bool,
        device=x.device,
    )


@signature("(..., m, n) -> bool[()]")
def is_tall(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor is tall along the given dimensions."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    return torch.tensor(
        x.shape[dim[0]] >= x.shape[dim[1]],
        dtype=torch.bool,
        device=x.device,
    )


@signature("(..., m, n) -> bool[()]")
def is_wide(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor is wide along the given dimensions."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    return torch.tensor(
        x.shape[dim[0]] <= x.shape[dim[1]],
        dtype=torch.bool,
        device=x.device,
    )


@signature("(..., n, n) -> bool[(...)]")
def is_identity(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is the identity."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    return torch.isclose(
        x,
        torch.eye(x.shape[-1], dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    ).all(dim=(-2, -1))


@signature("(..., m, n) -> bool[(...)]")
def is_boolean(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only zeros and ones."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    return ((x == 0) | (x == 1)).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_zero(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only zeros."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    return (x == 0).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_ones(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only ones."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    return (x == 1).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_one_hot(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor has exactly one 1 entry and zeros elsewhere."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    x = x.movedim(dim, (-2, -1))
    counts = x.sum(dim=(-2, -1))
    return is_boolean(x, dim=(-2, -1)) & (counts == 1)


@signature("(..., n, n) -> bool[(...)]")
def is_symmetric(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symmetric."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    return torch.isclose(
        x,
        x.swapaxes(*dim),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


@signature("(..., n, n) -> bool[(...)]")
def is_skew_symmetric(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is skew-symmetric."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    return torch.isclose(x, -x.swapaxes(*dim), rtol=rtol, atol=atol).all(dim=dim)


@signature("(..., n, n) -> bool[(...)]")
def is_orthogonal(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is orthogonal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    return torch.isclose(
        x @ x.swapaxes(*dim),
        torch.eye(x.shape[dim[-1]], device=x.device),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_column_orthogonal(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has orthonormal columns."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] < x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    gram = x.mT @ x
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.isclose(gram, eye, rtol=rtol, atol=atol).all(dim=(-2, -1))


@signature("(..., m, n) -> bool[(...)]")
def is_row_orthogonal(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has orthonormal rows."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] > x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    gram = x @ x.mT
    eye = torch.eye(x.shape[-2], dtype=x.dtype, device=x.device)
    return torch.isclose(gram, eye, rtol=rtol, atol=atol).all(dim=(-2, -1))


@signature("(..., m, n) -> bool[(...)]")
def is_row_stochastic(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether rows are nonnegative and sum to one."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    bounded = ((x >= 0) & (x <= 1)).all(dim=(-2, -1))
    row_sums = x.sum(dim=-1)
    ones = torch.ones_like(row_sums)
    return bounded & torch.isclose(row_sums, ones, rtol=rtol, atol=atol).all(dim=-1)


@signature("(..., m, n) -> bool[(...)]")
def is_row_centered(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether rows sum to zero."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    row_sums = x.sum(dim=-1)
    zeros = torch.zeros_like(row_sums)
    return torch.isclose(row_sums, zeros, rtol=rtol, atol=atol).all(dim=-1)


@signature("(..., m, n) -> bool[(...)]")
def is_column_stochastic(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether columns are nonnegative and sum to one."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    bounded = ((x >= 0) & (x <= 1)).all(dim=(-2, -1))
    col_sums = x.sum(dim=-2)
    ones = torch.ones_like(col_sums)
    return bounded & torch.isclose(col_sums, ones, rtol=rtol, atol=atol).all(dim=-1)


@signature("(..., m, n) -> bool[(...)]")
def is_column_centered(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether columns sum to zero."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    col_sums = x.sum(dim=-2)
    zeros = torch.zeros_like(col_sums)
    return torch.isclose(col_sums, zeros, rtol=rtol, atol=atol).all(dim=-1)


@signature("(..., m, n) -> bool[(...)]")
def is_doubly_centered(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether both rows and columns sum to zero."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    return is_row_centered(x, dim=dim, rtol=rtol, atol=atol) & is_column_centered(
        x, dim=dim, rtol=rtol, atol=atol
    )


@signature("(..., n, n) -> bool[(...)]")
def is_doubly_stochastic(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the matrix is square, row-stochastic, and column-stochastic."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    return (
        is_square(x, dim=dim)
        & is_row_stochastic(x, dim=dim, rtol=rtol, atol=atol)
        & is_column_stochastic(x, dim=dim, rtol=rtol, atol=atol)
    )


@signature("(..., n, n) -> bool[(...)]")
def is_permutation(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is a permutation matrix."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    return is_doubly_stochastic(x, dim=dim, rtol=rtol, atol=atol) & is_boolean(
        x, dim=dim
    )


@signature("(..., m, n) -> bool[(...)]")
def is_left_invertible(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has full column rank."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] < x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return ranks == x.shape[-1]


@signature("(..., m, n) -> bool[(...)]")
def is_right_invertible(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has full row rank."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] > x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return ranks == x.shape[-2]


@signature("(..., n, n) -> bool[(...)]")
def is_special_orthogonal(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is special orthogonal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)
    return is_orthogonal(x, rtol=rtol, atol=atol) & torch.isclose(
        torch.linalg.det(x),
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


@signature("(..., n, n) -> bool[(...)]")
def is_negative_diagonal(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is diagonal with strictly negative diagonal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1] or torch.is_complex(x):
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    diagonal = x.diagonal(dim1=-2, dim2=-1)
    return is_diagonal(x, dim=(-2, -1), rtol=rtol, atol=atol) & (diagonal < -atol).all(
        dim=-1
    )


@signature("(..., n, n) -> bool[(...)]")
def is_cholesky_factor(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is lower triangular with positive diagonal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1] or torch.is_complex(x):
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    diagonal = x.diagonal(dim1=-2, dim2=-1)
    return is_lower_triangular(x, dim=(-2, -1), rtol=rtol, atol=atol) & (
        diagonal > atol
    ).all(dim=-1)


@signature("(..., n, n) -> bool[(...)]")
def is_positive_diagonal(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is diagonal with strictly positive diagonal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1] or torch.is_complex(x):
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    diagonal = x.diagonal(dim1=-2, dim2=-1)
    return is_diagonal(x, dim=(-2, -1), rtol=rtol, atol=atol) & (diagonal > atol).all(
        dim=-1
    )


@signature("(..., n, n) -> bool[(...)]")
def is_positive_definite(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symmetric positive definite."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    symmetric = is_symmetric(x, dim=(-2, -1), rtol=rtol, atol=atol)
    eigenvalues = torch.linalg.eigvalsh(x)
    return symmetric & (eigenvalues > atol).all(dim=-1)


@signature("(..., n, n) -> bool[(...)]")
def is_positive_scalar_matrix(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor equals $σI$ for some scalar $σ > 0$."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)
    if x.shape[-1] == 0 or torch.is_complex(x):
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    diagonal = x.diagonal(dim1=-2, dim2=-1)
    constant_diagonal = torch.isclose(
        diagonal, diagonal[..., :1], rtol=rtol, atol=atol
    ).all(dim=-1)
    positive_scale = diagonal[..., 0] > atol

    return (
        is_diagonal(x, dim=(-2, -1), rtol=rtol, atol=atol)
        & constant_diagonal
        & positive_scale
    )


@signature("(..., n, n) -> bool[(...)]")
def is_positive_semidefinite(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symmetric positive semidefinite."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    symmetric = is_symmetric(x, dim=(-2, -1), rtol=rtol, atol=atol)
    eigenvalues = torch.linalg.eigvalsh(x)
    return symmetric & (eigenvalues >= 0).all(dim=-1)


@signature("(..., n, n) -> bool[(...)]")
def is_negative_definite(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symmetric negative definite."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    symmetric = is_symmetric(x, dim=(-2, -1), rtol=rtol, atol=atol)
    eigenvalues = torch.linalg.eigvalsh(x)
    return symmetric & (eigenvalues < -atol).all(dim=-1)


@signature("(..., n, n) -> bool[(...)]")
def is_negative_semidefinite(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symmetric negative semidefinite."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    symmetric = is_symmetric(x, dim=(-2, -1), rtol=rtol, atol=atol)
    eigenvalues = torch.linalg.eigvalsh(x)
    return symmetric & (eigenvalues <= 0).all(dim=-1)


@signature("(..., n, n) -> bool[(...)]")
def is_traceless(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Checks whether the trace of the given tensor is zero.

    Note:
        - Traceless matrices are an additive group.
        - Traceless relates to Lie-Algebras: <https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Lie_algebra>
        - In particular a complex matrix is traceless if and only if it is expressible as a commutator:
          tr(A) = 0 ⟺ ∑λᵢ = 0 ⟺ A = PQ-QP for some P,Q.
    """
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    return torch.isclose(
        torch.sum(x.diagonal(dim1=dim[-1], dim2=dim[-2]), dim=-1),
        torch.zeros((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )  # NOTE: no need for `all(dim=dim)` here


@signature("(..., n, n) -> bool[(...)]")
def is_normal(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is normal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    result = torch.isclose(
        x @ x.swapaxes(*dim), x.swapaxes(*dim) @ x, rtol=rtol, atol=atol
    )
    return result.all(dim=(-2, -1))


@signature("(..., 2n, 2n) -> bool[(...)]")
def is_symplectic(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symplectic."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    dim_x, dim_y = x.shape[-2], x.shape[-1]
    if dim_x != dim_y or dim_x % 2 != 0:
        raise ValueError("Expected square matrix of even size, got {x.shape}.")

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=x.device, dtype=x.dtype)
    eye = torch.eye(dim_x // 2, device=x.device, dtype=x.dtype)
    J = torch.kron(J1, eye)

    result = torch.isclose(x, J.T @ x @ J, rtol=rtol, atol=atol)
    return result.all(dim=(-2, -1))


@signature("(..., 2n, 2n) -> bool[(...)]")
def is_hamiltonian(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is Hamiltonian."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    dim_x, dim_y = x.shape[-2], x.shape[-1]
    if dim_x != dim_y or dim_x % 2 != 0:
        raise ValueError("Expected square matrix of even size, got {x.shape}.")

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=x.device, dtype=x.dtype)
    eye = torch.eye(dim_x // 2, device=x.device, dtype=x.dtype)
    J = torch.kron(J1, eye)

    # check if J @ x is symmetric
    return is_symmetric(J @ x, dim=(-2, -1), rtol=rtol, atol=atol)


# endregion matrix groups --------------------------------------------------------------


# region masked ------------------------------------------------------------------------
@signature("(..., m, n) -> bool[(...)]")
def is_diagonal(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is diagonal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    mask = torch.eye(x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    return is_masked(x, mask, rtol=rtol, atol=atol)


@signature("(..., m, n) -> bool[(...)]")
def is_lower_triangular(
    x: Tensor,
    /,
    lower: int = 0,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is lower triangular."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    return torch.isclose(x, x.tril(lower), rtol=rtol, atol=atol).all(dim=(-2, -1))


@signature("(..., m, n) -> bool[(...)]")
def is_upper_triangular(
    x: Tensor,
    /,
    upper: int = 0,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is lower triangular."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    return torch.isclose(x, x.triu(upper), rtol=rtol, atol=atol).all(dim=(-2, -1))


@signature("(..., m, n) -> bool[(...)]")
def is_triangular(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is lower or upper triangular."""
    return is_lower_triangular(
        x, shape=shape, dim=dim, rtol=rtol, atol=atol
    ) | is_upper_triangular(x, shape=shape, dim=dim, rtol=rtol, atol=atol)


@signature("(..., m, n) -> bool[(...)]")
def is_tridiagonal(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is tridiagonal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    return torch.isclose(x, x.triu(-1).tril(+1), rtol=rtol, atol=atol).all(dim=(-2, -1))


@signature("(..., m, n) -> bool[(...)]")
def is_block_diagonal(
    x: Tensor,
    /,
    block_sizes: tuple[int, ...] | None = None,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is block diagonal."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    if block_sizes is None:
        return torch.ones(x.shape[:-2], dtype=torch.bool, device=x.device)

    if not block_sizes or any(block_size <= 0 for block_size in block_sizes):
        raise ValueError("block_sizes must be a non-empty tuple of positive integers.")
    if sum(block_sizes) != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    mask = torch.block_diag(
        *(
            torch.ones(block_size, block_size, device=x.device, dtype=torch.bool)
            for block_size in block_sizes
        )
    )
    masked = torch.where(mask, x, 0)
    return torch.isclose(x, masked, rtol=rtol, atol=atol).all(dim=(-2, -1))


@signature("(..., m, n) -> bool[(...)]")
def is_toeplitz(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is constant along diagonals."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    return torch.isclose(x[..., 1:, 1:], x[..., :-1, :-1], rtol=rtol, atol=atol).all(
        dim=(-2, -1)
    )


@signature("(..., n, n) -> bool[(...)]")
def is_circulant(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is circulant."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)

    shifted_rows = torch.roll(x[..., :-1, :], shifts=1, dims=-1)
    return torch.isclose(x[..., 1:, :], shifted_rows, rtol=rtol, atol=atol).all(
        dim=(-2, -1)
    )


@signature("(..., m, n) -> bool[(...)]")
def is_banded(
    x: Tensor,
    /,
    lower: int,
    upper: int,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is banded."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    if not (lower <= 0 <= upper):
        raise ValueError("Lower bound must be greater than or equal to upper bound.")

    x = x.movedim(dim, (-2, -1))
    result = torch.isclose(x, x.triu(lower).tril(upper), rtol=rtol, atol=atol)
    return result.all(dim=(-2, -1))


@signature("(..., m, n) -> bool[(...)]")
def is_masked(
    x: Tensor,
    /,
    mask: Tensor,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is masked."""
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    mask_ = torch.as_tensor(mask, dtype=x.dtype, device=x.device)
    return torch.isclose(x, x * mask_, rtol=rtol, atol=atol).all(dim=dim)


# endregion masked checks --------------------------------------------------------------


# region other projections -------------------------------------------------------------
@signature("(..., n, n) -> bool[(...)]")
def is_projection(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is idempotent."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    x = x.movedim(dim, (-2, -1))
    if x.shape[-2] != x.shape[-1]:
        return torch.zeros(x.shape[:-2], dtype=torch.bool, device=x.device)
    return torch.isclose(x @ x, x, rtol=rtol, atol=atol).all(dim=(-2, -1))


@signature("(..., n, n) -> bool[(...)]")
def is_orthogonal_projection(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is a symmetric projection."""
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)
    return is_projection(x, dim=dim, rtol=rtol, atol=atol) & is_symmetric(
        x, dim=dim, rtol=rtol, atol=atol
    )


@signature("(..., m, n) -> bool[(...)]")
def is_spectral_normalized(
    x: Tensor,
    /,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor has unit spectral norm.

    .. math:: (‖A‖₂-1) ≤ rtol⋅𝟏 + atol

    This is done by checking whether the spectral norm is less than or equal to L.
    """
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    # TODO: compute spectral norm with given tolerance
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=dim)
    return (sigma - 1.0) <= (1.0 + rtol) * 1.0 + atol


@signature("(..., m, n) -> bool[(...)]")
def is_lipschitz_bounded(
    x: Tensor,
    /,
    lipschitz_bound: float,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor has bounded lipschitz constant.

    .. math:: ‖A‖₂ ≤ (1+rtol)⋅L + atol

    This is done by checking whether the spectral norm is less than or equal to L.
    """
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    # TODO: compute spectral norm with given tolerance
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=dim)
    return sigma <= (1 + rtol) * lipschitz_bound + atol


@signature("(..., m, n) -> bool[(...)]")
def is_contraction(
    x: Tensor,
    /,
    lipschitz_bound: float = 1.0,
    shape: tuple[int, int] | None = None,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is a contraction.

    This is done by checking whether the spectral norm is less than or equal to 1.
    If strict, we require that the spectral norm is strictly less than 1, more specifically
    we include tolerance:

    .. math:: ‖A‖₂ ≤ (1-rtol)⋅𝟏 - atol
    """
    if shape is not None and not _has_shape(x, shape, dim):
        return _full_false(x, dim)
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=dim)
    one = torch.ones_like(sigma)
    c = torch.full_like(sigma, lipschitz_bound)
    return sigma <= torch.minimum(one, ((1.0 + rtol) * c + atol))


@signature("(..., m, n) -> bool[(...)]")
def is_diagonally_dominant(
    x: Tensor,
    /,
    size: int | None = None,
    *,
    strict: bool = False,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given matrix is diagonally dominant.

    The test is based on the definition of diagonally dominant matrices:

    .. math:: Aᵢᵢ ≥ ∑_{j≠i} |Aᵢⱼ| \quad\text{for all \(i = 1, …, n\)}

    If `strict=True`, we require that the inequality is strict for all $i$, more specifically
    we include tolerance:

    .. math:: Aᵢᵢ ≥ (1+\text{rtol})⋅(∑_{j≠i} |Aᵢⱼ|) + \text{atol} \quad\text{for all \(i = 1, …, n\)}

    In this case, the matrix is guaranteed to be invertible
    """
    if size is not None and not _has_size(x, size, dim):
        return _full_false(x, dim)

    m, n = dim
    if x.shape[m] != x.shape[n]:
        raise ValueError("Expected square matrix")

    x_abs = x.abs()
    lhs = 2 * x_abs.diagonal(dim1=m, dim2=n)  # (*BS, n)
    rhs = x_abs.movedim(dim, (-2, -1)).sum(dim=-1)  # (*BS, n)

    if strict:
        return (lhs >= ((1 + rtol) * rhs + atol)).all(dim=-1)
    return (lhs >= rhs).all(dim=-1)


@signature("(..., m, n) -> bool[(...)]")
def is_forward_stable(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given matrix is forward stable.

    Note:
        An m×n matrix A is forward stable if and only if $𝐄[Aᵢⱼ] = 0$ and $𝐕[Aᵢⱼ] = 1/n$
    """
    num = x.shape[dim[-1]]
    mean = x.mean(dim=dim)
    stdv = x.std(dim=dim)
    zeros = torch.zeros_like(mean)
    ones = torch.ones_like(stdv)
    mean_stable = torch.isclose(mean, zeros, atol=atol, rtol=rtol)
    stdv_stable = torch.isclose(stdv, ones / num, atol=atol, rtol=rtol)
    return mean_stable & stdv_stable


@signature("(..., m, n) -> bool[(...)]")
def is_backward_stable(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given matrix is backward stable.

    Note:
        An m×n matrix A is backward stable if and only if $𝐄[Aᵢⱼ] = 0$ and $𝐕[Aᵢⱼ] = 1/m$
    """
    return is_forward_stable(x.swapaxes(*dim), dim=dim, rtol=rtol, atol=atol)


# endregion other projections ----------------------------------------------------------
# endregion is_* checks ----------------------------------------------------------------
