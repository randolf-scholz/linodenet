r"""Dense layers."""

__all__ = [
    "ReverseDense",
]

from typing import Final, Optional

from torch import Tensor, nn

from signatures import signature

from .activations import Activation, get_activation


class ReverseDense(nn.Module):
    r"""Linear layer with pre-activation rather than post activation $x ⟼ A⋅ϕ(x) + b$.

    It has been shown that doing the activation first leads to consistently better performance in
    ResNets. Note that due to the residual structure, this is equivalent in expressiveness
    to post-activation.

    References:
        - | Identity Mappings in Deep Residual Networks
          | He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
          | European Conference on Computer Vision 2016
          | https://doi.org/10.1007/978-3-319-46493-0_38
    """

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

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "bias": self.bias is not None,
            "activation": self.activation,
        }

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        activation: str | Activation,
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
    def forward(self, x: Tensor, /) -> Tensor:
        return self.linear(self.activation(x))
