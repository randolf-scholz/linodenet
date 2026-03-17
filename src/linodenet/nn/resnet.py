r"""Residual Network Implementation."""

__all__ = [
    # Classes
    "ResNet",
    "ResNetBlock",
]

from torch import Tensor, nn

from linodenet.nn.containers import ModuleSequence
from linodenet.nn.reverse_dense import ReverseDense
from linodenet.nn.rezero import ReZero
from signatures import signature


class ResNetBlock(ModuleSequence):
    r"""Pre-activation ResNet block.

    References:
        - | Identity Mappings in Deep Residual Networks
          | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
          | European Conference on Computer Vision 2016
          | https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """

    def __init__(
        self,
        input_size: int,
        *,
        num_layers: int = 2,
        use_rezero: bool = True,
        activation: str = "relu",
        use_batchnorm: bool = False,
    ) -> None:
        layers: list[nn.Module] = []

        for _ in range(num_layers):
            layer = ReverseDense(
                input_size=input_size,
                output_size=input_size,
                activation=activation,
            )
            layers.append(layer)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(input_size))
        if use_rezero:
            layers.append(ReZero())

        super().__init__(layers)


class ResNet(ModuleSequence[ResNetBlock]):
    r"""A ResNet model."""

    def __init__(self, input_size: int, *, num_block: int = 5) -> None:
        blocks: list[ResNetBlock] = []
        for _ in range(num_block):
            block = ResNetBlock(input_size=input_size)
            blocks.append(block)

        super().__init__(blocks)

    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        for block in self:
            x = x + block(x)
        return x
