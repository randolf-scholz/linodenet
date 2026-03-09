r"""Implementations of the regularized gelu activation function."""

__all__ = ["reglu", "ReGLU"]

import torch
from torch import Tensor, nn

from signatures import signature


@signature("[(..., *ds), (..., *ds)] -> (..., *ds)")
def reglu(a: Tensor, b: Tensor) -> Tensor:
    r"""Regularized gelu activation function.

    .. math:: ReGLU( (a, b) ) = a ⊙ relu(b)

    >>> x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    >>> reglu(x, x)
    tensor([-0., 0., 1., 4.])

    References:
        - | Shazeer, Noam.
          | “GLU Variants Improve Transformer.”
          | arXiv:2002.05202. Preprint, arXiv, February 12, 2020.
          | https://doi.org/10.48550/arXiv.2002.05202.
    """
    return a * torch.relu(b)


class ReGLU(nn.Module):
    r"""ReLU-GLU activation function.

    .. math:: ReGLU( (a, b) ) = a ⊙ relu(b)

    >>> act = ReGLU()
    >>> x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    >>> act(x, x)
    tensor([-0., 0., 1., 4.])

    References:
        - | Shazeer, Noam.
          | “GLU Variants Improve Transformer.”
          | arXiv:2002.05202. Preprint, arXiv, February 12, 2020.
          | https://doi.org/10.48550/arXiv.2002.05202.
    """

    @signature("[(..., *ds), (..., *ds)] -> (..., *ds)")
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return reglu(a, b)
