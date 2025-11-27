r"""Linear Filter Cells."""

__all__ = [
    "LinearCell",
    "LinearResidualCell",
]

from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn

from linodenet.filters.base import CellBase


class LinearCell(CellBase):
    r"""Linear RNN Cell.

    .. math:: F(y，x) =  Ux + Vy + b

    where $U$ and $V$ are learnable matrices, and $b$ is a learnable bias vector.
    """

    # PARAMETERS
    U: Tensor
    r"""PARAM: the hidden state matrix."""
    V: Tensor
    r"""PARAM: the observable matrix."""
    bias: Optional[Tensor]
    r"""PARAM: the bias vector."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__(input_size, hidden_size)
        n = self.input_size
        m = self.hidden_size
        self.U = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.V = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        self.bias = nn.Parameter(torch.zeros(m)) if bool(bias) else None

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the cell.

        .. math:: F(y，x) =  Ux + Vy + b

        .. Signature:: ``[(..., n), (..., m)] -> (..., m)``.
        """
        z = torch.einsum("ij, ...i -> ...j", self.U, x)
        z = z + torch.einsum("ij, ...i -> ...j", self.V, y)

        if self.bias is not None:
            z = z + self.bias
        return z


class LinearResidualCell(CellBase):
    r"""Linear RNN Cell that performs a residual update.

    .. math:: x' = x - F⋅(Hy - x)

    Where $F$ is a learnable square matrix, and $H$ is either a learnable matrix or
    a fixed matrix.
    """

    # PARAMETERS
    F: Tensor
    r"""PARAM: the hidden state matrix."""
    H: Optional[Tensor]
    r"""PARAM: the observable matrix."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        m, n = self.hidden_size, self.input_size
        self.F = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.H = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r = torch.einsum("...i,ij->...j", y, self.H) - x
        return x - torch.einsum("...i,ij->...j", r, self.F)
