r"""Dense layers."""

__all__ = [
    "ReverseDense",
]

from typing import Final, Optional

from torch import Tensor, nn

from linodenet.activations import Activation, get_activation
from linodenet.signatures import signature


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
        nn.init.kaiming_uniform_(self.weight, nonlinearity=activation_name)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias[None], nonlinearity=activation_name)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    @signature("(..., m) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.activation(x))
