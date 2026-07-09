r"""Residual Network Implementation."""

__all__ = ["ResNet"]

from typing import Final, Optional

from torch import Tensor, nn

from signatures import signature

from .activations import Activations
from .containers import ModuleSequence
from .rezero import ReZero


class ResNet(ModuleSequence[nn.Module]):
    r"""A ResNet model.

    References:
        - | Identity Mappings in Deep Residual Networks
          | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
          | European Conference on Computer Vision 2016
          | https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """

    input_size: Final[int]
    r"""CONST: Input and output dimensionality."""
    num_blocks: Final[int]
    r"""CONST: Number of residual blocks."""
    layers_per_block: Final[int]
    r"""CONST: Number of layers in each residual block."""
    latent_size: Final[int]
    r"""CONST: Hidden size used inside each residual block."""
    use_rezero: Final[bool]
    r"""CONST: Whether to wrap blocks in ``ReZero``."""
    use_batchnorm: Final[bool]

    def __init__(
        self,
        input_size: int,
        *,
        num_blocks: int = 5,
        num_block: int | None = None,
        layers_per_block: int = 2,
        latent_size: int | None = None,
        activation: str | nn.Module = "relu",
        use_rezero: bool = True,
        scalar_map: Optional[nn.Module] = None,
        use_batchnorm: bool = False,
    ) -> None:
        self.input_size = input_size
        self.num_blocks = num_blocks if num_block is None else num_block
        self.layers_per_block = layers_per_block
        self.latent_size = input_size if latent_size is None else latent_size
        self.use_rezero = use_rezero
        self.use_batchnorm = use_batchnorm

        if self.layers_per_block < 1:
            raise ValueError("layers_per_block must be at least 1")
        if self.latent_size != self.input_size and self.layers_per_block <= 1:
            raise ValueError(
                "latent_size must equal input_size when layers_per_block <= 1"
            )
        if scalar_map is not None and not self.use_rezero:
            raise ValueError("scalar_map requires use_rezero=True")

        blocks = [
            self._make_block(
                input_size=self.input_size,
                layers_per_block=self.layers_per_block,
                latent_size=self.latent_size,
                activation=activation,
                use_rezero=self.use_rezero,
                scalar_map=scalar_map,
                use_batchnorm=use_batchnorm,
            )
            for _ in range(self.num_blocks)
        ]
        super().__init__(blocks)

    @staticmethod
    def _make_dense_layer(
        input_size: int,
        output_size: int,
        /,
        *,
        activation: str | nn.Module,
        use_batchnorm: bool,
    ) -> list[nn.Module]:
        linear = nn.Linear(input_size, output_size)
        if isinstance(activation, str):
            nn.init.kaiming_uniform_(linear.weight, nonlinearity=activation)  # type: ignore[arg-type]
            if linear.bias is not None:
                nn.init.kaiming_uniform_(linear.bias[None], nonlinearity=activation)  # type: ignore[arg-type]

        layers: list[nn.Module] = [Activations.new(activation), linear]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(output_size))
        return layers

    @staticmethod
    def _make_block(
        *,
        input_size: int,
        layers_per_block: int,
        latent_size: int,
        activation: str | nn.Module,
        use_rezero: bool,
        scalar_map: nn.Module | None,
        use_batchnorm: bool,
    ) -> nn.Module:
        layers: list[nn.Module] = []
        if layers_per_block == 1:
            layers.extend(
                ResNet._make_dense_layer(
                    input_size,
                    input_size,
                    activation=activation,
                    use_batchnorm=use_batchnorm,
                )
            )
        else:
            layers.extend(
                ResNet._make_dense_layer(
                    input_size,
                    latent_size,
                    activation=activation,
                    use_batchnorm=use_batchnorm,
                )
            )
            layers.extend(
                module
                for _ in range(layers_per_block - 2)
                for module in ResNet._make_dense_layer(
                    latent_size,
                    latent_size,
                    activation=activation,
                    use_batchnorm=use_batchnorm,
                )
            )
            layers.extend(
                ResNet._make_dense_layer(
                    latent_size,
                    input_size,
                    activation=activation,
                    use_batchnorm=use_batchnorm,
                )
            )

        block = nn.Sequential(*layers)
        if use_rezero:
            return ReZero(block, scalar_map=scalar_map)
        return block

    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor, /) -> Tensor:
        for block in self:
            x = x + block(x)
        return x
