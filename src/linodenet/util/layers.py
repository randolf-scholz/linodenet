r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "ReZero",
    "ReZeroCell",
    "ReverseDense",
]

from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.util._util import deep_dict_update, initialize_from_config


class ReZeroCell(nn.Module):
    r"""ReZero module.

    Simply multiplies the inputs by a scalar initialized to zero.
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
    }
    r"""The hyperparameter dictionary"""

    # PARAMETERS
    scalar: Tensor
    r"""The scalar to multiply the inputs by."""

    def __init__(self, scalar: Optional[Tensor] = None) -> None:
        super().__init__()
        if scalar is None:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.scalar = scalar

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
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
        r"""Forward pass.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        for k, block in enumerate(self):
            x = x + self.weights[k] * block(x)
        return x


class ReverseDense(nn.Module):
    r"""ReverseDense module $x→A⋅ϕ(x)$."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
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

    def __init__(self, input_size: int, output_size: int, **HP: Any) -> None:
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        self.input_size = HP["input_size"] = input_size
        self.output_size = HP["output_size"] = output_size

        self.activation: nn.Module = initialize_from_config(HP["activation"])

        self.linear = nn.Linear(HP["input_size"], HP["output_size"], HP["bias"])
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        activation_name = HP["activation"]["__name__"].lower()
        nn.init.kaiming_uniform_(self.weight, nonlinearity=activation_name)

        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias[None], nonlinearity=activation_name)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass."""
        return self.linear(self.activation(x))
