r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "ReZero",
    "ReverseDense",
]

import logging
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.util._util import autojit, deep_dict_update, initialize_from_config

__logger__ = logging.getLogger(__name__)


@autojit
class ReZero(nn.Module):
    """ReZero module.

    Simply multiplies the inputs by a scalar intitialized to zero.
    """

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
    }
    r"""The hyperparameter dictionary"""

    # PARAMETERS
    scalar: Tensor
    r"""The scalar to multiply the inputs by."""

    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(0.0))

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


@autojit
class ReverseDense(nn.Module):
    """ReverseDense module `x→A⋅ϕ(x)`."""

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "output_size": None,
        "activation": {
            "__name__": "ReLU",
            "__module__": "torch.nn",
            "inplace": False,
        },
    }
    r"""The hyperparameter dictionary"""

    input_size: Final[int]
    """The size of the input"""
    output_size: Final[int]
    """The size of the output"""

    # PARAMETERS
    weight: Tensor
    r"""The weight matrix."""
    bias: Optional[Tensor]
    r"""The bias vector."""

    def __init__(self, **HP: Any) -> None:
        super().__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        self.input_size = HP["input_size"]
        self.output_size = HP["output_size"]

        self.activation = initialize_from_config(HP["activation"])

        self.linear = nn.Linear(HP["input_size"], HP["output_size"])
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        activation_name = HP["activation"]["__name__"].lower()
        nn.init.kaiming_uniform_(self.weight, nonlinearity=activation_name)

        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias[None], nonlinearity=activation_name)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass."""
        return self.linear(self.activation(x))
