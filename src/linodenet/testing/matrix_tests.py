r"""Checks for testing certain matrix properties (type (1,1)-tensors)."""

__all__ = [
    # ABCs & Protocols
    "MatrixTest",
    # is_* checks
    "is_backward_stable",
    "is_banded",
    "is_contraction",
    "is_diagonal",
    "is_diagonally_dominant",
    "is_forward_stable",
    "is_hamiltonian",
    "is_low_rank",
    "is_lower_triangular",
    "is_masked",
    "is_normal",
    "is_orthogonal",
    "is_rank_one",
    "is_skew_symmetric",
    "is_square",
    "is_symmetric",
    "is_symplectic",
    "is_traceless",
    "is_tridiagonal",
    "is_upper_triangular",
]

from typing import Protocol

import torch
from torch import Tensor

from linodenet.constants import ATOL, RTOL
from signatures import signature


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


# region is_* checks -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
@signature("(..., m, n) -> bool[(...)]")
def is_low_rank(
    x: Tensor,
    /,
    rank: int,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is low-rank."""
    # move target dims to -1 and -2
    x = x.movedim(dim, (-2, -1))
    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return ranks <= rank


@signature("(..., m, n) -> bool[(...)]")
def is_rank_one(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is rank-1."""
    # move target dims to -1 and -2
    x = x.movedim(dim, (-2, -1))
    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return ranks <= 1


@signature("(..., m, n) -> bool[()]")
def is_square(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor is square along the given dimensions."""
    return torch.tensor(
        x.shape[dim[0]] == x.shape[dim[1]],
        dtype=torch.bool,
        device=x.device,
    )


@signature("(..., n, n) -> bool[(...)]")
def is_symmetric(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symmetric."""
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
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is skew-symmetric."""
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
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is orthogonal."""
    return torch.isclose(
        x @ x.swapaxes(*dim),
        torch.eye(x.shape[dim[-1]], device=x.device),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


@signature("(..., n, n) -> bool[(...)]")
def is_traceless(
    x: Tensor,
    /,
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
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is normal."""
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
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symplectic."""
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")

    m, n = dim
    dim_x, dim_y = x.shape[m], x.shape[n]
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
    ).all(dim=dim)


@signature("(..., 2n, 2n) -> bool[(...)]")
def is_hamiltonian(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is Hamiltonian."""
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")

    m, n = dim
    dim_x, dim_y = x.shape[m], x.shape[n]
    if dim_x != dim_y or dim_x % 2 != 0:
        raise ValueError("Expected square matrix of even size, got {x.shape}.")

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=x.device, dtype=x.dtype)
    eye = torch.eye(dim_x // 2, device=x.device, dtype=x.dtype)
    J = torch.kron(J1, eye)

    # check if J @ x is symmetric
    return is_symmetric(J @ x, dim=dim, rtol=rtol, atol=atol)


# endregion matrix groups --------------------------------------------------------------


# region masked ------------------------------------------------------------------------
@signature("(..., m, n) -> bool[(...)]")
def is_diagonal(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is diagonal."""
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")

    m, n = dim
    mask = torch.eye(x.shape[m], x.shape[n], device=x.device, dtype=x.dtype)
    return torch.isclose(
        x,
        x * mask,
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_lower_triangular(
    x: Tensor,
    /,
    lower: int = 0,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is lower triangular."""
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")

    return torch.isclose(x, x.tril(lower), rtol=rtol, atol=atol).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_upper_triangular(
    x: Tensor,
    /,
    upper: int = 0,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is lower triangular."""
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")

    return torch.isclose(
        x,
        x.triu(upper),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_tridiagonal(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is tridiagonal."""
    return torch.isclose(
        x,
        x.triu(-1).tril(+1),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_banded(
    x: Tensor,
    /,
    lower: int,
    upper: int,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is banded."""
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")
    if not (lower <= 0 <= upper):
        raise ValueError("Lower bound must be greater than or equal to upper bound.")

    return torch.isclose(
        x,
        x.triu(lower).tril(upper),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


@signature("(..., m, n) -> bool[(...)]")
def is_masked(
    x: Tensor,
    /,
    mask: Tensor,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is masked."""
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
def is_contraction(
    x: Tensor,
    /,
    *,
    strict: bool = False,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is a contraction.

    This is done by checking whether the spectral norm is less than or equal to 1.
    If strict, we require that the spectral norm is strictly less than 1, more specifically
    we include tolerance:

    .. math:: σ(A) ≤ (1-rtol)⋅r - atol
    """
    # TODO: compute spectral norm with given tolerance
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=dim)
    if strict:
        return sigma <= ((1.0 - rtol) * 1.0 - atol)
    return sigma <= 1.0


@signature("(..., m, n) -> bool[(...)]")
def is_diagonally_dominant(
    x: Tensor,
    /,
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
