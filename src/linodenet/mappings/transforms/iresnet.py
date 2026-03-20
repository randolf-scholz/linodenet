r"""Simple i-ResNet built from residual contraction blocks."""

__all__ = ["IResNet"]

from typing import Final, Optional

from torch import nn

from linodenet.mappings.base import TransformSequence
from linodenet.mappings.linear import LinearContraction
from linodenet.mappings.transforms.residual import (
    ResidualContraction,
    ReZeroContraction,
)
from linodenet.nn.activations import get_activation


class IResNet(TransformSequence[ResidualContraction | ReZeroContraction]):
    r"""Invertible residual network built from contractive residual blocks.

    References:
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | http://proceedings.mlr.press/v97/behrmann19a.html
        - https://github.com/jhjacobsen/invertible-resnet
    """

    input_size: Final[int]
    r"""CONST: Input and output dimensionality."""
    num_blocks: Final[int]
    r"""CONST: Number of residual blocks."""
    layers_per_block: Final[int]
    r"""CONST: Number of linear layers in each residual block."""
    latent_size: Final[int]
    r"""CONST: Hidden size used inside each residual block."""
    use_rezero: Final[bool]
    r"""CONST: Whether to wrap blocks in ``ReZeroContraction``."""

    def __init__(
        self,
        input_size: int,
        *,
        num_blocks: int,
        layers_per_block: int,
        latent_size: int | None = None,
        activation: str | nn.Module = "ReLU",
        use_rezero: bool = False,
        scalar_map: Optional[nn.Module | str] = None,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        self.input_size = input_size
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        self.latent_size = input_size if latent_size is None else latent_size
        self.use_rezero = use_rezero

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
                maxiter=maxiter,
                atol=atol,
                rtol=rtol,
            )
            for _ in range(self.num_blocks)
        ]
        super().__init__(blocks)

    @staticmethod
    def _make_block(
        *,
        input_size: int,
        layers_per_block: int,
        latent_size: int,
        activation: str | nn.Module,
        use_rezero: bool,
        scalar_map: nn.Module | str | None,
        maxiter: int,
        atol: float,
        rtol: float,
    ) -> ResidualContraction | ReZeroContraction:
        layers: list[nn.Module] = []
        act = get_activation(activation)
        assert isinstance(act, nn.Module)
        if layers_per_block < 1:
            raise ValueError("layers_per_block must be at least 1")
        if layers_per_block == 1:
            layers.extend([LinearContraction(input_size, input_size), act])
        else:
            layers.extend([LinearContraction(input_size, latent_size), act])
            layers.extend(
                module
                for _ in range(layers_per_block - 2)
                for module in (LinearContraction(latent_size, latent_size), act)
            )
            layers.extend([LinearContraction(latent_size, input_size), act])

        contraction = nn.Sequential(*layers)
        if use_rezero:
            return ReZeroContraction(
                contraction,
                scalar_map=scalar_map,
                maxiter=maxiter,
                atol=atol,
                rtol=rtol,
            )
        return ResidualContraction(
            contraction,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
        )
