r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "ReZero",
    "ReZeroCell",
    "ReverseDense",
]

from collections.abc import Mapping

import torch
from torch import Tensor, jit, nn
from typing_extensions import Any, Final, Optional, Self

from linodenet.constants import EMPTY_MAP
from linodenet.utils._utils import deep_dict_update, initialize_from_dict


class ReZeroCell(nn.Module):
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
        self, *, scalar: Optional[Tensor] = None, learnable: bool = True
    ) -> None:
        super().__init__()
        self.learnable = learnable

        initial_value = torch.as_tensor(0.0 if scalar is None else scalar)
        self.scalar = nn.Parameter(initial_value) if self.learnable else initial_value

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(...,) -> (...,)``."""
        return self.scalar * x


class ReZero(nn.Sequential):
    r"""A ReZero model."""

    def __init__(self, *blocks: nn.Module, weights: Optional[Tensor] = None) -> None:
        super().__init__()

        if weights is None:
            self.weights = nn.Parameter(torch.zeros(len(blocks)))
        else:
            self.weights = nn.Parameter(weights)
        super().__init__(*blocks)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        for k, block in enumerate(self):
            x = x + self.weights[k] * block(x)
        return x


class ReverseDense(nn.Module):
    r"""ReverseDense module $x ⟼ A⋅ϕ(x) + b$."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "output_size": None,
        "bias": True,
        "activation": {
            "__name__": "ReLU",
            "__module__": "torch.nn",
            "inplace": False,
        },
    }
    r"""The hyperparameter dictionary"""

    input_size: Final[int]
    r"""The size of the input"""
    output_size: Final[int]
    r"""The size of the output"""

    # PARAMETERS
    weight: Tensor
    r"""The weight matrix."""
    bias: Optional[Tensor]
    r"""The bias vector."""

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] = EMPTY_MAP, /, **kwargs: Any) -> Self:
        """Initialize from hyperparameters."""
        config = cls.HP | dict(cfg, **kwargs)
        return cls(**config)

    def __init__(self, input_size: int, output_size: int, **cfg: Any) -> None:
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        self.input_size = config["input_size"] = input_size
        self.output_size = config["output_size"] = output_size

        self.activation: nn.Module = initialize_from_dict(config["activation"])

        self.linear = nn.Linear(
            config["input_size"], config["output_size"], config["bias"]
        )
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        activation_name = config["activation"]["__name__"].lower()
        nn.init.kaiming_uniform_(self.weight, nonlinearity=activation_name)

        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias[None], nonlinearity=activation_name)

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m) -> (..., n)``."""
        return self.linear(self.activation(x))
