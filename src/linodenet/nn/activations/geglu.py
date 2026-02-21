r"""Gated GeLU activation function."""

__all__ = [
    "geglu",
    "GEGLU",
]

import torch.nn.functional as F
from torch import Tensor, nn

from linodenet.signatures import signature


@signature("[(..., d), (..., d)] -> (..., d)")
def geglu(a: Tensor, b: Tensor) -> Tensor:
    r"""GEGLU activation function.

    .. math:: GeGLU( (a, b) ) = a ⊙ gelu(b)

    >>> geglu(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    tensor([-1.9545,  0.0000])

    References:
        - | Shazeer, Noam.
          | “GLU Variants Improve Transformer.”
          | arXiv:2002.05202. Preprint, arXiv, February 12, 2020.
          | https://doi.org/10.48550/arXiv.2002.05202.
    """
    return a * F.gelu(b)


class GEGLU(nn.Module):
    r"""GEGLU activation function with learnable parameters.

    .. math:: GEGLU( (a, b) ) = a ⊙ gelu(b)

    >>> act = GEGLU()
    >>> act(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    tensor([-1.9545,  0.0000])

    References:
        - | Shazeer, Noam.
          | “GLU Variants Improve Transformer.”
          | arXiv:2002.05202. Preprint, arXiv, February 12, 2020.
          | https://doi.org/10.48550/arXiv.2002.05202.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)
