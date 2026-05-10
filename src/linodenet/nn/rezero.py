r"""ReZero layers.

References:
    - | ReZero is all you need: fast convergence at large depth.
      | Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, Julian McAuley
      | Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, PMLR
      | https://proceedings.mlr.press/v161/bachlechner21a.html
"""

__all__ = [
    "ReZero",
    "ReZeroResNet",
    "resolve_gate",
]

import warnings
from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor, nn

from signatures import signature


class ReZero[
    M: nn.Module = nn.Module,
    S: nn.Module = nn.Module,
](nn.Module):
    r"""ReZero module.

    Simply multiplies the inputs by a scalar initialized to zero.
    """

    scalar: Tensor
    r"""PARAM: The scalar to multiply the inputs by."""
    scalar_map: S
    r"""MODULE: Map applied to the scalar before scaling the input."""
    module: M
    r"""MODULE: Map applied to the inputs before scaling them."""

    @property
    def config(self) -> dict:
        return {
            "module": self.module,
            "scalar": self.scalar,
            "scalar_map": self.scalar_map,
        }

    def __init__[U: nn.Module = nn.Identity, V: nn.Module = nn.Identity](
        self: ReZero[U, V],
        module: U | None = None,
        *,
        scalar_map: V | None = None,
        initial_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(initial_value))
        self.module = cast("U", nn.Identity() if module is None else module)
        self.scalar_map = cast("V", nn.Identity() if scalar_map is None else scalar_map)

    @signature("(..., *xs) -> (..., *xs)")
    def forward(self, x: Tensor) -> Tensor:
        return self.scalar_map(self.scalar) * self.module(x)


class ReZeroResNet(nn.ModuleList):
    r"""A Residual Network with ReZero scalars."""

    @property
    def config(self) -> dict:
        return {"modules": list(self)}

    def __init__(self, modules: Iterable[nn.Module]) -> None:
        module_list = list(modules)

        for i, module in enumerate(module_list):
            # pass if already a ReZeroCell
            if isinstance(module, ReZero):
                continue
            module_list[i] = ReZero(module)

        super().__init__(module_list)

    def forward(self, x: Tensor, /) -> Tensor:
        for block in self:
            x = x + block(x)
        return x


def resolve_gate(
    gate: str | nn.Module | None,
    /,
    *,
    scalar_map: nn.Module | None = None,
) -> nn.Module:
    match gate:
        case "rezero":
            return ReZero(scalar_map=scalar_map)
        case None | "identity":
            if scalar_map is not None:
                warnings.warn(
                    "Ignoring scalar_map because gate is not 'rezero'.",
                    stacklevel=3,
                )
            return nn.Identity()
        case nn.Module():
            if scalar_map is not None:
                warnings.warn(
                    "Ignoring scalar_map because gate is not 'rezero'.",
                    stacklevel=3,
                )
            return gate
        case str():
            raise ValueError(
                f"Unknown gate: {gate!r}. "
                "Expected 'rezero', 'identity', None, or an nn.Module."
            )
        case _:
            raise TypeError(
                f"gate must be a string, nn.Module, or None, got {type(gate)!r}."
            )
