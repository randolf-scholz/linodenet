r"""Fallback implementations of matrix functions via eigendecomposition."""

__all__ = [
    "matrix_log",
    "matrix_sqrt",
]

import torch
from torch import Tensor


def matrix_log(x: Tensor, /) -> Tensor:
    r"""Compute the principal matrix logarithm via eigendecomposition."""
    eigenvalues, eigenvectors = torch.linalg.eig(x)
    transformed = torch.diag_embed(eigenvalues.log())
    return eigenvectors @ transformed @ torch.linalg.inv(eigenvectors)


def matrix_sqrt(x: Tensor, /) -> Tensor:
    r"""Compute the symmetric positive-semidefinite square root.

    Note:
        This function assumes that `x` is symmetric/Hermitian positive semidefinite.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(x)
    transformed = torch.diag_embed(eigenvalues.clamp_min(0).sqrt())
    sqrt_matrix = eigenvectors @ transformed @ eigenvectors.mH
    return 0.5 * (sqrt_matrix + sqrt_matrix.mH)
