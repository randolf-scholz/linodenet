r"""Residual Network Implementation.

Modified variant of the implementation from https://github.com/yandex-research/rtdl

Original Licensed under Apache License 2.0
"""

__all__ = [
    # Classes
    "ResNet",
    "ResNetBlock",
]

from torch import Tensor, jit, nn

from linodenet.containers import ModuleSequence
from linodenet.layers import (
    ReverseDense,
    ReZero,
)
from linodenet.signatures import signature


class ResNetBlock(ModuleSequence):
    r"""Pre-activation ResNet block.

    References:
        Identity Mappings in Deep Residual Networks
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        European Conference on Computer Vision 2016
        https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
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

    @jit.export
    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        for block in self:
            x = x + block(x)
        return x
