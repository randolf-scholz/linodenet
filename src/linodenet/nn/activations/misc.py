r"""Miscellaneous activation functions."""

__all__ = ["EntLU", "entlu"]

import torch
from torch import Tensor, nn


def entlu(x: Tensor) -> Tensor:
    one_m_x = torch.where(x > 0, torch.zeros_like(x), 1 - x)
    return torch.where(x > 0, x + 1, torch.exp(torch.special.entr(one_m_x)))


class EntLU(nn.Module):
    r"""Maps tensor elementwise via $x ↦ ⟦x>0 ? x + 1 : eᴴ⁽¹⁻ˣ⁾⟧$."""

    def forward(self, x: Tensor) -> Tensor:
        return entlu(x)
