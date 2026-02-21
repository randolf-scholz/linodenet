r"""Implementations of the regularized gelu activation function."""

__all__ = ["reglu", "ReGLU"]

import torch
from torch import Tensor, nn


def reglu(x: Tensor) -> Tensor:
    r"""Regularized gelu activation function.

    .. math:: ReGLU( (a, b) ) = a ⊙ relu(b)

    >>> reglu(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    tensor([-1.,  0.])

    References:
        - | Shazeer, Noam.
          | “GLU Variants Improve Transformer.”
          | arXiv:2002.05202. Preprint, arXiv, February 12, 2020.
          | https://doi.org/10.48550/arXiv.2002.05202.
    """
    a, b = x.chunk(2, dim=-1)
    return a * torch.relu(b)


class ReGLU(nn.Module):
    r"""ReLU-GLU activation function.

    .. math:: ReGLU( (a, b) ) = a ⊙ relu(b)

    >>> act = ReGLU()
    >>> act(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    tensor([-1.,  0.])

    References:
        - | Shazeer, Noam.
          | “GLU Variants Improve Transformer.”
          | arXiv:2002.05202. Preprint, arXiv, February 12, 2020.
          | https://doi.org/10.48550/arXiv.2002.05202.
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)
