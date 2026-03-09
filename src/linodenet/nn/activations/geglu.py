r"""Gated GeLU activation function."""

__all__ = [
    "geglu",
    "GEGLU",
]

import torch
import torch.nn.functional as F
from torch import Tensor

from signatures import signature


@signature("[(..., *ds), (..., *ds)] -> (..., *ds)")
def geglu(a: Tensor, b: Tensor) -> Tensor:
    r"""GEGLU activation function.

    .. math:: GeGLU( (a, b) ) = a ⊙ gelu(b)

    >>> x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    >>> geglu(x, x)
    tensor([0.1587, 0.0000, 0.8413, 3.9090])

    References:
        - | Shazeer, Noam.
          | “GLU Variants Improve Transformer.”
          | arXiv:2002.05202. Preprint, arXiv, February 12, 2020.
          | https://doi.org/10.48550/arXiv.2002.05202.
    """
    return a * F.gelu(b)


class GEGLU(torch.nn.Module):
    r"""GEGLU activation function with learnable parameters.

    .. math:: GEGLU( (a, b) ) = a ⊙ gelu(b)

    >>> act = GEGLU()
    >>> x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    >>> act(x, x)
    tensor([0.1587, 0.0000, 0.8413, 3.9090])

    References:
        - | Shazeer, Noam.
          | “GLU Variants Improve Transformer.”
          | arXiv:2002.05202. Preprint, arXiv, February 12, 2020.
          | https://doi.org/10.48550/arXiv.2002.05202.
    """

    @signature("[(..., *ds), (..., *ds)] -> (..., *ds)")
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return geglu(a, b)
