r"""Fallback implementations of matrix functions via eigendecomposition."""

__all__ = [
    "matrix_log",
    "matrix_sqrt",
]

import torch
from torch import Tensor

from signatures import signature


@signature("(..., d, d) -> (..., d, d)")
def matrix_log(x: Tensor, /) -> Tensor:
    r"""Compute the principal matrix logarithm via eigendecomposition."""
    eigenvalues, V = torch.linalg.eig(x)
    log_evals = eigenvalues.log()
    # compute X = VDV⁻¹ ⟺ XV = VD
    return torch.linalg.solve(
        V,
        torch.einsum("...ij, ...j -> ...ij", V, log_evals),
        left=False,
    )


@signature("(..., d, d) -> (..., d, d)")
def matrix_sqrt(x: Tensor, /) -> Tensor:
    r"""Compute the symmetric positive-semidefinite square root.

    Note:
        This function assumes that `x` is symmetric/Hermitian positive semidefinite.
    """
    eigenvalues, V = torch.linalg.eigh(x)
    sqrt_evals = eigenvalues.clamp_min(0).sqrt()
    sqrt_matrix = torch.einsum("...ik,...k,...jk->...ij", V, sqrt_evals, V.conj())
    return 0.5 * (sqrt_matrix + sqrt_matrix.mH)
