"""Checks for testing certain matrix properties (type (1,1)-tensors)."""

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
    "is_skew_symmetric",
    "is_symmetric",
    "is_symplectic",
    "is_traceless",
    "is_upper_triangular",
]

import torch
from torch import Tensor
from typing_extensions import Protocol

from linodenet.constants import ATOL, ONE, RTOL, TRUE, ZERO


class MatrixTest(Protocol):
    r"""Protocol for testing certain matrix property."""

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

        .. Signature:: ``(..., m, n) -> bool[...]``

        Note:
            - There are different kinds of matrix groups, which are not cleanly separated
              here at the moment (additive, multiplicative, Lie groups, ...).
            - JIT does not support positional-only and keyword-only arguments.
              So they are only used in the protocol.
        """
        ...


# region is_* checks -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
def is_symmetric(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symmetric.

    .. Signature:: ``(..., n, n) -> bool[...]``
    """
    return torch.isclose(
        x,
        x.swapaxes(*dim),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


def is_skew_symmetric(
    x: Tensor,
    *,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is skew-symmetric.

    .. Signature:: ``(..., n, n) -> bool[...]``
    """
    return torch.isclose(
        x,
        -x.swapaxes(*dim),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


def is_low_rank(
    x: Tensor,
    rank: int = 1,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is low-rank.

    .. Signature:: ``(..., m, n) -> bool[...]``
    """
    ranks = torch.linalg.matrix_rank(x, rtol=rtol, atol=atol)
    return (ranks <= rank).all(dim=dim)


def is_orthogonal(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is orthogonal.

    .. Signature:: ``(..., n, n) -> bool[...]``
    """
    return torch.isclose(
        x @ x.swapaxes(*dim),
        torch.eye(x.shape[dim[-1]], device=x.device),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


def is_traceless(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    """Checks whether the trace of the given tensor is zero.

    .. Signature:: ``(..., n, n) -> bool[...]``

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


def is_normal(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is normal.

    .. Signature:: ``(..., n, n) -> bool[...]``
    """
    return torch.isclose(
        x @ x.swapaxes(*dim),
        x.swapaxes(*dim) @ x,
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


def is_symplectic(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is symplectic.

    .. Signature:: ``(..., 2n, 2n) -> bool[...]``
    """
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


def is_hamiltonian(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is Hamiltonian.

    .. Signature:: ``(..., 2n, 2n) -> bool``
    """
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
def is_diagonal(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is diagonal.

    .. Signature:: ``(..., m, n) -> bool[...]``
    """
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


def is_lower_triangular(
    x: Tensor,
    lower: int = 0,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is lower triangular.

    .. Signature:: ``(..., m, n) -> bool[...]``
    """
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")

    return torch.isclose(x, x.tril(lower), rtol=rtol, atol=atol).all(dim=dim)


def is_upper_triangular(
    x: Tensor,
    upper: int = 0,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is lower triangular.

    .. Signature:: ``(..., m, n) -> bool[...]``
    """
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")

    return torch.isclose(
        x,
        x.triu(upper),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


def is_banded(
    x: Tensor,
    upper: int = 0,
    lower: int = 0,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is banded.

    .. Signature:: ``(..., m, n) -> bool[...]``
    """
    if dim != (-2, -1):
        raise NotImplementedError("Currently only supports dim=(-2,-1).")

    return torch.isclose(
        x,
        x.tril(lower).triu(upper),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


def is_masked(
    x: Tensor,
    mask: Tensor = TRUE,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Tensor:
    r"""Check whether the given tensor is masked.

    .. Signature:: ``(..., m, n) -> bool[...]``
    """
    mask_ = torch.as_tensor(mask, dtype=x.dtype, device=x.device)
    return torch.isclose(
        x,
        x * mask_,
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


# endregion masked checks --------------------------------------------------------------


# region other projections -------------------------------------------------------------
def is_contraction(
    x: Tensor,
    strict: bool = False,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is a contraction.

    .. Signature:: ``(..., m, n) -> bool``

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


def is_diagonally_dominant(
    x: Tensor,
    strict: bool = False,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given matrix is diagonally dominant.

    .. Signature:: ``(..., n, n) -> bool``

    The test is based on the definition of diagonally dominant matrices:

    .. math:: Aᵢᵢ ≥ ∑_{j≠i} |Aᵢⱼ| for all i = 1, …, n

    If strict, we require that the inequality is strict for all i, more specifically
    we include tolerance:

    .. math:: Aᵢᵢ ≥ (1+rtol)⋅(∑_{j≠i} |Aᵢⱼ|) + atol for all i = 1, …, n

    Note:
        Strictly diagonally dominant matrices are invertible.
    """
    m, n = dim

    if x.shape[m] != x.shape[n]:
        raise ValueError("Expected square matrix")

    x_abs = x.abs()
    lhs = 2 * x_abs.diagonal(dim1=m, dim2=n)
    rhs = x_abs.sum(dim=-1)

    if strict:
        return lhs >= ((1 + rtol) * rhs + atol)
    return lhs >= rhs


def is_forward_stable(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    """Check whether the given matrix is forward stable.

    Note:
        An m×n matrix A is forward stable if and only if $𝐄[Aᵢⱼ] = 0$ and $𝐕[Aᵢⱼ] = 1/n$
    """
    N = x.shape[dim[-1]]
    mean = x.mean(dim=dim)
    stdv = x.std(dim=dim)
    mean_stable = torch.isclose(mean, ZERO, atol=atol, rtol=rtol).all(dim=dim)
    stdv_stable = torch.isclose(stdv, ONE / N, atol=atol, rtol=rtol).all(dim=dim)
    return mean_stable & stdv_stable


def is_backward_stable(
    x: Tensor,
    dim: tuple[int, int] = (-2, -1),
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    """Check whether the given matrix is backward stable.

    Note:
        An m×n matrix A is backward stable if and only if $𝐄[Aᵢⱼ] = 0$ and $𝐕[Aᵢⱼ] = 1/m$
    """
    return is_forward_stable(x.swapaxes(*dim), dim=dim, rtol=rtol, atol=atol)


# endregion other projections ----------------------------------------------------------
# endregion is_* checks ----------------------------------------------------------------
