"""Checks for testing if tensor belongs to matrix group."""

__all__ = [
    "is_banded",
    "is_diagonal",
    "is_masked",
    "is_normal",
    "is_orthogonal",
    "is_skew_symmetric",
    "is_symmetric",
    "is_traceless",
]


import torch
from torch import BoolTensor, Tensor, jit

from linodenet.constants import TRUE


# region is_* checks -------------------------------------------------------------------
@jit.script
def is_symmetric(x: Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check whether the given tensor is symmetric.

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    return torch.allclose(
        x,
        x.swapaxes(-1, -2),
        rtol=rtol,
        atol=atol,
    )


@jit.script
def is_skew_symmetric(x: Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check whether the given tensor is skew-symmetric.

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    return torch.allclose(
        x,
        -x.swapaxes(-1, -2),
        rtol=rtol,
        atol=atol,
    )


@jit.script
def is_normal(x: Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check whether the given tensor is normal.

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    return torch.allclose(
        x @ x.swapaxes(-1, -2),
        x.swapaxes(-1, -2) @ x,
        rtol=rtol,
        atol=atol,
    )


@jit.script
def is_traceless(x: Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """Checks whether the trace of the given tensor is zero.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

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
def is_orthogonal(x: Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check whether the given tensor is orthogonal.

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    return torch.allclose(
        x @ x.swapaxes(-1, -2),
        torch.eye(x.shape[-1], device=x.device),
        rtol=rtol,
        atol=atol,
    )


@jit.script
def is_diagonal(x: Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check whether the given tensor is diagonal.

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    return torch.allclose(
        x,
        x * torch.eye(x.shape[-1], device=x.device),
        rtol=rtol,
        atol=atol,
    )


@jit.script
def is_banded(
    x: Tensor, upper: int = 0, lower: int = 0, rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    r"""Check whether the given tensor is banded.

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    return torch.allclose(
        x,
        torch.tril(torch.triu(x, diagonal=upper), diagonal=lower),
        rtol=rtol,
        atol=atol,
    )


@jit.script
def is_masked(
    x: Tensor, m: BoolTensor = TRUE, rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    r"""Check whether the given tensor is masked.

    .. Signature:: ``(..., n, n) -> (..., n, n)``
    """
    return torch.allclose(
        x,
        x * torch.as_tensor(m, dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


# endregion is_* checks ----------------------------------------------------------------
