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

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
    }

    value: Tensor
    r"""BUFFER: The constant value."""

    def __init__(self, value: float | Tensor) -> None:
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value))

    @signature("() -> (...)")
    def forward(self) -> Tensor:
        return self.value


class Identity(nn.Module):
    r"""Identity with HP attribute."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
    }
    r"""Hyperparameters of the component."""

    @signature("(...) -> (...)")
    def forward(self, x: Tensor) -> Tensor:
        return x
