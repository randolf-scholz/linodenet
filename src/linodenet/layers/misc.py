r"""Miscellaneous layers."""

__all__ = [
    "Constant",
    "Identity",
]

import torch
from torch import Tensor, nn

from linodenet.signatures import signature


class Constant(nn.Module):
    r"""Constant function."""

    value: Tensor
    r"""BUFFER: The constant value."""

    @property
    def config(self) -> dict:
        return {
            "value": self.value,
        }

    def __init__(self, value: float | Tensor) -> None:
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value))

    @signature("() -> (...)")
    def forward(self) -> Tensor:
        return self.value


class Identity(nn.Module):
    r"""Identity layer."""

    @property
    def config(self) -> dict:
        return {}

    @signature("(...) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        return x
