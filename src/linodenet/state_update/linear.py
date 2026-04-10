r"""Linear filters."""

__all__ = [
    "LinearCell",
    "LinearInnovationCell",
]

from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from linodenet.nn import ReZero
from signatures import signature

from .base import StateUpdaterBase


class LinearCell(StateUpdaterBase):
    r"""Linear state update.

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
        m = self.hidden_size
        n = self.input_size
        self.U = nn.Parameter(torch.normal(0, 1 / sqrt(m), size=(m, m)))
        self.V = nn.Parameter(torch.normal(0, 1 / sqrt(n), size=(m, n)))
        self.bias = nn.Parameter(torch.zeros(m)) if bool(bias) else None

    @signature("[(..., n), (..., m)] -> (..., m)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the state update.

        .. math:: F(y，x) =  Ux + Vy + b
        """
        return F.linear(x, self.U, None) + F.linear(y, self.V, self.bias)


class LinearInnovationCell(StateUpdaterBase):
    r"""Linear innovation state update.

    .. math:: x' = x - ρ(K(y - h(x)))

    where $K$ is a learnable innovation gain, $h$ is the observation map, and
    $ρ$ is a gate applied to the innovation correction.

    Standard gate options are:

    - ``"rezero"``: use a learnable ReZero scalar $ρ(z)=αz$ with $α$ initialized
      to zero, so that the cell starts as the identity map.
    - ``"identity"``: use $ρ(z)=z$ with no additional scaling.
    - ``nn.Module``: use a custom user-provided gate.

    The observation map can be:

    - ``"linear"``: use a learned linear observation map.
    - ``"identity"``: use $h(x)=x$, which requires ``input_size == hidden_size``.
    - ``nn.Module``: use a custom user-provided observation map.
    """

    # PARAMETERS
    gain: nn.Linear
    r"""MODULE: The learnable innovation gain."""
    observation_map: nn.Module
    r"""MODULE: The observation map used in the innovation term."""
    scalar: Tensor | None
    r"""PARAM: Optional scalar exposed by the gate."""
    gate: nn.Module
    r"""MODULE: Optional gate for the innovation term."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        gate: str | nn.Module = "rezero",
        observation_map: str | nn.Module = "linear",
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)

        self.gain = nn.Linear(input_size, hidden_size, bias=False)

        match gate:
            case nn.Module():
                self.gate = gate
                self.scalar = getattr(gate, "scalar", None)
            case "rezero":
                self.gate = ReZero()
                self.scalar = self.gate.scalar
            case "identity":
                self.gate = nn.Identity()
                self.scalar = None
            case str():
                raise ValueError(
                    f"Unknown gate: {gate!r}. "
                    "Expected 'rezero', 'identity', or an nn.Module."
                )
            case _:
                raise TypeError(
                    f"gate must be a string or nn.Module, got {type(gate)!r}."
                )

        match observation_map:
            case nn.Module():
                self.observation_map = observation_map
            case "linear":
                self.observation_map = nn.Linear(hidden_size, input_size, bias=False)
            case "identity":
                if input_size != hidden_size:
                    raise ValueError(
                        "observation_map='identity' requires input_size == hidden_size!"
                    )
                self.observation_map = nn.Identity()
            case str():
                raise ValueError(
                    f"Unknown observation_map: {observation_map!r}. "
                    "Expected 'linear', 'identity', or an nn.Module."
                )
            case _:
                raise TypeError(
                    "observation_map must be a string or nn.Module, "
                    f"got {type(observation_map)!r}."
                )

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r = y - self.observation_map(x)
        return x - self.gate(self.gain(r))
