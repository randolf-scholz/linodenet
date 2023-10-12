"""Checks for testing if tensor belongs to matrix group."""

__all__ = [
    # Protocol
    "MatrixTest",
    # is_* checks
    "is_hamiltonian",
    "is_normal",
    "is_lowrank",
    "is_orthogonal",
    "is_skew_symmetric",
    "is_symmetric",
    "is_symplectic",
    "is_traceless",
    # masked
    "is_banded",
    "is_diagonal",
    "is_lower_triangular",
    "is_masked",
    "is_upper_triangular",
]


import warnings
from typing import Protocol

import torch
from torch import BoolTensor, Tensor, jit

from linodenet.constants import ATOL, RTOL, TRUE


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
@jit.script
def is_symmetric(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is symmetric.

    .. Signature:: ``(..., n, n) -> bool``
    """
    return torch.allclose(x, x.swapaxes(-1, -2), rtol=rtol, atol=atol)


@jit.script
def is_skew_symmetric(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is skew-symmetric.

    .. Signature:: ``(..., n, n) -> bool``
    """
    return torch.allclose(x, -x.swapaxes(-1, -2), rtol=rtol, atol=atol)


@jit.script
def is_lowrank(
    x: Tensor, rank: int = 1, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is low-rank.

    .. Signature:: ``(..., m, n) -> bool``
    """
    return bool((torch.linalg.matrix_rank(x, rtol=rtol, atol=atol) <= rank).all())


@jit.script
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


@jit.script
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


@jit.script
def is_normal(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is normal.

    .. Signature:: ``(..., n, n) -> bool``
    """
    return torch.allclose(
        x @ x.swapaxes(-1, -2), x.swapaxes(-1, -2) @ x, rtol=rtol, atol=atol
    )


@jit.script
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


@jit.script
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
@jit.script
def is_diagonal(x: Tensor, rtol: float = RTOL, atol: float = ATOL) -> bool:
    r"""Check whether the given tensor is diagonal.

    .. Signature:: ``(..., m, n) -> bool``
    """
    mask = torch.eye(x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.allclose(x, x * mask, rtol=rtol, atol=atol)


@jit.script
def is_lower_triangular(
    x: Tensor, lower: int = 0, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is lower triangular.

    .. Signature:: ``(..., m, n) -> bool``
    """
    return torch.allclose(x, x.tril(lower), rtol=rtol, atol=atol)


@jit.script
def is_upper_triangular(
    x: Tensor, upper: int = 0, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is lower triangular.

    .. Signature:: ``(..., m, n) -> bool``
    """
    return torch.allclose(x, x.triu(upper), rtol=rtol, atol=atol)


@jit.script
def is_banded(
    x: Tensor, upper: int = 0, lower: int = 0, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is banded.

    .. Signature:: ``(..., m, n) -> bool``
    """
    return torch.allclose(x, x.tril(lower).triu(upper), rtol=rtol, atol=atol)


@jit.script
def is_masked(
    x: Tensor, mask: BoolTensor = TRUE, rtol: float = RTOL, atol: float = ATOL
) -> bool:
    r"""Check whether the given tensor is masked.

    .. Signature:: ``(..., m, n) -> bool``
    """
    mask_ = torch.as_tensor(mask, dtype=x.dtype, device=x.device)
    return torch.allclose(x, x * mask_, rtol=rtol, atol=atol)


# endregion masked checks --------------------------------------------------------------

# endregion is_* checks ----------------------------------------------------------------
