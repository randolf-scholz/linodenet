"""Checks for testing if tensor belongs to matrix group."""

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


import warnings

import torch
from torch import BoolTensor, Tensor
from typing_extensions import Protocol

from linodenet.constants import ATOL, ONE, RTOL, TRUE, ZERO


class MatrixTest(Protocol):
    r"""Protocol for testing if matrix belongs to matrix group."""

    def __call__(self, x: Tensor, /, *, rtol: float = RTOL, atol: float = ATOL) -> bool:
        r"""Check whether the given matrix belongs to a matrix group/manifold.

        .. Signature:: ``(..., m, n) -> bool``

        Args:
            x: Matrix to be tested.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Note:
            - There are different kinds of matrix groups, which are not cleanly separated
              here at the moment (additive, multiplicative, Lie groups, ...).
            - JIT does not support positional-only and keyword-only arguments.
              So they are only used in the protocol.
        """
        ...


# region is_* checks -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
def is_symmetric(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is symmetric.

    .. Signature:: ``(..., n, n) -> bool``
    """
    return torch.allclose(x, x.swapaxes(-1, -2), rtol=rtol, atol=atol)


def is_skew_symmetric(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is skew-symmetric.

    .. Signature:: ``(..., n, n) -> bool``
    """
    return torch.allclose(x, -x.swapaxes(-1, -2), rtol=rtol, atol=atol)


def is_low_rank(
    x: Tensor, rank: int = 1, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is low-rank.

    .. Signature:: ``(..., m, n) -> bool``
    """
    return bool((torch.linalg.matrix_rank(x, rtol=rtol, atol=atol) <= rank).all())


def is_orthogonal(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is orthogonal.

    .. Signature:: ``(..., n, n) -> bool``
    """
    return torch.allclose(
        x @ x.swapaxes(-1, -2),
        torch.eye(x.shape[-1], device=x.device),
        rtol=rtol,
        atol=atol,
    )


def is_traceless(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    """Checks whether the trace of the given tensor is zero.

    .. Signature:: ``(..., n, n) -> bool``

    Note:
        - Traceless matrices are an additive group.
        - Traceless relates to Lie-Algebras: <https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Lie_algebra>
        - In particular a complex matrix is traceless if and only if it is expressible as a commutator:
          tr(A) = 0 ⟺ ∑λᵢ = 0 ⟺ A = PQ-QP for some P,Q.
    """
    return torch.allclose(
        torch.diagonal(x, dim1=-1, dim2=-2).sum(dim=-1),
        torch.zeros(x.shape[:-2], dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


def is_normal(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is normal.

    .. Signature:: ``(..., n, n) -> bool``
    """
    return torch.allclose(
        x @ x.swapaxes(-1, -2), x.swapaxes(-1, -2) @ x, rtol=rtol, atol=atol
    )


def is_symplectic(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is symplectic.

    .. Signature:: ``(..., 2n, 2n) -> bool``
    """
    m, n = x.shape[-2:]
    if m != n or m % 2 != 0:
        warnings.warn(
            f"Expected square matrix of even size, but got {x.shape}.",
            stacklevel=2,
        )
        return False

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=x.device, dtype=x.dtype)
    eye = torch.eye(m // 2, device=x.device, dtype=x.dtype)
    J = torch.kron(J1, eye)

    return torch.allclose(x, J.T @ x @ J, rtol=rtol, atol=atol)


def is_hamiltonian(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is Hamiltonian.

    .. Signature:: ``(..., 2n, 2n) -> bool``
    """
    m, n = x.shape[-2:]
    if m != n or m % 2 != 0:
        warnings.warn(
            f"Expected square matrix of even size, but got {x.shape}.",
            stacklevel=2,
        )
        return False

    # create J matrix
    J1 = torch.tensor([[0, 1], [-1, 0]], device=x.device, dtype=x.dtype)
    eye = torch.eye(m // 2, device=x.device, dtype=x.dtype)
    J = torch.kron(J1, eye)

    # check if J @ x is symmetric
    return is_symmetric(J @ x, rtol=rtol, atol=atol)


# endregion matrix groups --------------------------------------------------------------


# region masked ------------------------------------------------------------------------
def is_diagonal(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is diagonal.

    .. Signature:: ``(..., m, n) -> bool``
    """
    mask = torch.eye(x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.allclose(x, x * mask, rtol=rtol, atol=atol)


def is_lower_triangular(
    x: Tensor, lower: int = 0, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is lower triangular.

    .. Signature:: ``(..., m, n) -> bool``
    """
    return torch.allclose(x, x.tril(lower), rtol=rtol, atol=atol)


def is_upper_triangular(
    x: Tensor, upper: int = 0, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is lower triangular.

    .. Signature:: ``(..., m, n) -> bool``
    """
    return torch.allclose(x, x.triu(upper), rtol=rtol, atol=atol)


def is_banded(
    x: Tensor, upper: int = 0, lower: int = 0, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is banded.

    .. Signature:: ``(..., m, n) -> bool``
    """
    return torch.allclose(x, x.tril(lower).triu(upper), rtol=rtol, atol=atol)


def is_masked(
    x: Tensor, mask: BoolTensor = TRUE, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is masked.

    .. Signature:: ``(..., m, n) -> bool``
    """
    mask_ = torch.as_tensor(mask, dtype=x.dtype, device=x.device)
    return torch.allclose(x, x * mask_, rtol=rtol, atol=atol)


# endregion masked checks --------------------------------------------------------------


# region other projections -------------------------------------------------------------
def is_contraction(
    x: Tensor,
    strict: bool = False,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    r"""Check whether the given tensor is a contraction.

    .. Signature:: ``(..., m, n) -> bool``

    This is done by checking whether the spectral norm is less than or equal to 1.
    If strict, we require that the spectral norm is strictly less than 1, more specifically
    we include tolerance:

    .. math:: σ(A) ≤ (1-rtol)⋅r - atol
    """
    # TODO: compute spectral norm with given tolerance
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=(-2, -1))
    if strict:
        return bool((sigma <= ((1 - rtol) * 1.0 - atol)).all().item())
    return bool((sigma <= 1.0).all().item())


def is_diagonally_dominant(
    x: Tensor,
    strict: bool = False,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
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
    m, n = x.shape[-2:]
    if m != n:
        raise ValueError("Expected square matrix")

    x_abs = x.abs()
    lhs = 2 * x_abs.diagonal(dim1=-2, dim2=-1)
    rhs = x_abs.sum(dim=-1)

    if strict:
        return bool((lhs >= (1 + rtol) * rhs + atol).all())
    return bool((lhs >= rhs).all())


def is_forward_stable(
    x: Tensor,
    *,
    dims: tuple[int, int] = (-2, -1),
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    """Check whether the given matrix is forward stable.

    Note:
        An m×n matrix A is forward stable if and only if $𝐄[Aᵢⱼ] = 0$ and $𝐕[Aᵢⱼ] = 1/n$
    """
    n = x.shape[dims[-1]]
    mean = x.mean(dim=(-2, -1))
    stdv = x.std(dim=(-2, -1))

    return (
        torch.allclose(mean, ZERO, atol=atol, rtol=rtol)
        and torch.allclose(stdv, ONE / n, atol=atol, rtol=rtol)
    )  # fmt: skip


def is_backward_stable(
    x: Tensor,
    *,
    dims: tuple[int, int] = (-2, -1),
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    """Check whether the given matrix is backward stable.

    Note:
        An m×n matrix A is backward stable if and only if $𝐄[Aᵢⱼ] = 0$ and $𝐕[Aᵢⱼ] = 1/m$
    """
    m = x.shape[dims[-2]]
    mean = x.mean(dim=(-2, -1))
    stdv = x.std(dim=(-2, -1))

    return (
        torch.allclose(mean, ZERO, atol=atol, rtol=rtol)
        and torch.allclose(stdv, ONE / m, atol=atol, rtol=rtol)
    )  # fmt: skip


# endregion other projections ----------------------------------------------------------
# endregion is_* checks ----------------------------------------------------------------
