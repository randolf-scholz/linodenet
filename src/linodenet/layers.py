r"""Some layers and modules for neural networks."""

__all__ = [
    # Classes
    "Constant",
    "ReZero",
    "ReZeroResNet",
    "ReverseDense",
]

from collections.abc import Iterable, Mapping
from typing import Any, Final, Optional, Self

import torch
from torch import Tensor, jit, nn

from linodenet.activations import Activation, get_activation
from linodenet.constants import EMPTY_MAP


class Constant(nn.Module):
    r"""Constant function."""

    value: Tensor
    r"""BUFFER: The constant value."""

    def __init__(self, value: float | Tensor) -> None:
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value))

    def forward(self, _: Tensor) -> Tensor:
        return self.value


class ReZero(nn.Module):
    r"""ReZero module.

    Simply multiplies the inputs by a scalar initialized to zero.
    """

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
    }
    r"""The hyperparameter dictionary"""

    # CONSTANTS
    learnable: Final[bool]
    r"""CONST: Whether the scalar is learnable."""

    # PARAMETERS
    scalar: Tensor
    r"""The scalar to multiply the inputs by."""

    def __init__(
        self,
        module: Optional[nn.Module] = None,
        *,
        scalar: Optional[Tensor] = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        initial_value = torch.as_tensor(0.0 if scalar is None else scalar)
        self.scalar = nn.Parameter(initial_value) if self.learnable else initial_value
        self.learnable = learnable
        self.module = module

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(...,) -> (...,)``."""
        if self.module is None:
            return self.scalar * x
        return self.scalar * self.module(x)


class ReZeroResNet(nn.ModuleList):
    r"""A Residual Network with ReZero scalars."""

    def __init__(self, modules: Iterable[nn.Module]) -> None:
        module_list = list(modules)

        for i, module in enumerate(module_list):
            # pass if already a ReZeroCell
            if isinstance(module, ReZero):
                continue
            module_list[i] = ReZero(module)

        super().__init__(module_list)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        for block in self:
            x = x + block(x)
        return x


class ReverseDense(nn.Module):
    r"""ReverseDense module $x ⟼ A⋅ϕ(x) + b$."""

    input_size: Final[int]
    r"""The size of the input"""
    output_size: Final[int]
    r"""The size of the output"""

    # PARAMETERS
    activation: Activation
    r"""The activation function to apply after the linear transformation."""
    weight: Tensor
    r"""The weight matrix."""
    bias: Optional[Tensor]
    r"""The bias vector."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": int,
        "output_size": int,
        "bias": True,
        "activation": {
            "__name__": "ReLU",
            "__module__": "torch.nn",
            "inplace": False,
        },
    }
    r"""The hyperparameter dictionary"""

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] = EMPTY_MAP, /, **kwargs: Any) -> Self:
        r"""Initialize from hyperparameters."""
        config = cls.HP | dict(cfg, **kwargs)
        return cls(**config)  # type: ignore[arg-type]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        activation: str | Activation | type[Activation],
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(self.input_size, self.output_size, bias=bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        # initialize activation
        self.activation = get_activation(activation)
        activation_name = self.activation.__class__.__name__.lower()
        nn.init.kaiming_uniform_(self.weight, nonlinearity=activation_name)

        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias[None], nonlinearity=activation_name)

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m) -> (..., n)``."""
        return self.linear(self.activation(x))
