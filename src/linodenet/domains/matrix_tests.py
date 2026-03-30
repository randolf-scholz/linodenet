r"""Checks for testing certain matrix properties (type (1,1)-tensors)."""

__all__ = [
    # ABCs & Protocols
    "MatrixTest",
    "MatrixTestWithArgs",
    # is_* checks
    "is_backward_stable",
    "is_banded",
    "is_contraction",
    "is_column_orthogonal",
    "is_spectral_normalized",
    "is_lipschitz_bounded",
    "is_left_invertible",
    "is_diagonal",
    "is_diagonally_dominant",
    "is_forward_stable",
    "is_hamiltonian",
    "is_identity",
    "is_low_rank",
    "is_lower_triangular",
    "is_masked",
    "is_negative_definite",
    "is_negative_semidefinite",
    "is_normal",
    "is_orthogonal",
    "is_positive_definite",
    "is_positive_semidefinite",
    "is_right_invertible",
    "is_row_orthogonal",
    "is_special_orthogonal",
    "is_rank_one",
    "is_skew_symmetric",
    "is_square",
    "is_symmetric",
    "is_symplectic",
    "is_traceless",
    "is_tridiagonal",
    "is_upper_triangular",
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
    return torch.isclose(
        x,
        -x.swapaxes(*dim),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


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
        torch.diagonal(x, dim1=dim[-1], dim2=dim[-2]).sum(dim=-1),
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
    return torch.isclose(
        x @ x.swapaxes(*dim),
        x.swapaxes(*dim) @ x,
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


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

    return torch.isclose(
        x,
        J.T @ x @ J,
        rtol=rtol,
        atol=atol,
    ).all(dim=(-2, -1))


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
    return torch.isclose(
        x,
        x * mask,
        rtol=rtol,
        atol=atol,
    ).all(dim=(-2, -1))


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
    return torch.isclose(
        x,
        x.triu(upper),
        rtol=rtol,
        atol=atol,
    ).all(dim=(-2, -1))


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
    return torch.isclose(
        x,
        x.triu(-1).tril(+1),
        rtol=rtol,
        atol=atol,
    ).all(dim=(-2, -1))


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
    return torch.isclose(
        x,
        x.triu(lower).tril(upper),
        rtol=rtol,
        atol=atol,
    ).all(dim=(-2, -1))


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
    return torch.isclose(
        x,
        x * mask_,
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


# endregion masked checks --------------------------------------------------------------


# region other projections -------------------------------------------------------------
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
    mean_stable = torch.isclose(mean, zeros, atol=atol, rtol=rtol).all(dim=dim)
    stdv_stable = torch.isclose(stdv, ones / num, atol=atol, rtol=rtol).all(dim=dim)
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
