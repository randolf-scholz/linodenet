r"""Miscellaneous activation functions."""

__all__ = ["EntLU", "entlu"]

import torch
from torch import Tensor, nn


def entlu(x: Tensor) -> Tensor:
    one_m_x = torch.where(x > 0, 0.0, 1 - x)
    return torch.where(x > 0, x + 1, torch.exp(torch.special.entr(one_m_x)))


class EntLU(nn.Module):
    r"""Map tensor entries via the entropic linear unit.

    $$ ϕ(x)=⟦x>0 ? x+1 : \exp(H(1-x))⟧ $$

    Here $H(t)=-t\log(t)$ is the entropy primitive for $t>0$.
    The map preserves tensor shape and has strictly positive outputs.
    """

    def forward(self, x: Tensor) -> Tensor:
        return entlu(x)
