r"""ReZero layers.

References:
    - | ReZero is all you need: fast convergence at large depth.
      | Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, Julian McAuley
      | Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, PMLR
      | https://proceedings.mlr.press/v161/bachlechner21a.html
"""

__all__ = [
    "ReZero",
    "resolve_gate",
]

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
        initial_value: Tensor | float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.scalar = nn.Parameter(
            torch.as_tensor(initial_value), requires_grad=learnable
        )
        self.module = cast("U", nn.Identity() if module is None else module)
        self.scalar_map = cast("V", nn.Identity() if scalar_map is None else scalar_map)

    @signature("(..., *xs) -> (..., *xs)")
    def forward(self, x: Tensor) -> Tensor:
        return self.scalar_map(self.scalar) * self.module(x)

    @signature("(..., *xs) -> (..., *xs)")
    def right_inverse(self, y: Tensor) -> Tensor | None:
        if getattr(self.module, "right_inverse", None) is None:
            return None

        return self.module.right_inverse(y / self.scalar_map(self.scalar))  # type: ignore[operator]


def resolve_gate(gate: str | nn.Module | None, /) -> nn.Module:
    match gate:
        case None | "identity":
            return nn.Identity()

        case "rezero":
            return ReZero()

        case nn.Module():
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
