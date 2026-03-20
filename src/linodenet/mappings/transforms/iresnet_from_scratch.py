r"""Simple i-ResNet built from residual contraction blocks."""

__all__ = ["IResNet"]

from typing import Final

from torch import nn

from linodenet.mappings.base import TransformSequence
from linodenet.mappings.linear import LinearContraction
from linodenet.mappings.transforms.residual import (
    ResidualContraction,
    ReZeroContraction,
)
from linodenet.nn.activations import get_activation


class IResNet(TransformSequence[ResidualContraction | ReZeroContraction]):
    r"""Invertible residual network built from contractive residual blocks."""

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

        blocks = [
            self._make_block(
                input_size=self.input_size,
                layers_per_block=self.layers_per_block,
                latent_size=self.latent_size,
                activation=activation,
                use_rezero=self.use_rezero,
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
        maxiter: int,
        atol: float,
        rtol: float,
    ) -> ResidualContraction | ReZeroContraction:
        layers: list[nn.Module] = []
        if layers_per_block < 1:
            raise ValueError("layers_per_block must be at least 1")
        elif layers_per_block == 1:
            layers.extend(
                [
                    LinearContraction(input_size, input_size),
                    get_activation(activation),
                ]
            )
        else:
            layers.extend(
                [
                    LinearContraction(input_size, latent_size),
                    get_activation(activation),
                ]
            )
            layers.extend(
                module
                for _ in range(layers_per_block - 2)
                for module in (
                    LinearContraction(latent_size, latent_size),
                    get_activation(activation),
                )
            )
            layers.extend(
                [
                    LinearContraction(latent_size, input_size),
                    get_activation(activation),
                ]
            )

        contraction = nn.Sequential(*layers)
        if use_rezero:
            return ReZeroContraction(
                contraction,
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
