r"""Residual flow built from bottleneck contraction blocks."""

__all__ = ["BottleneckFlow"]

import warnings
from typing import Final

from torch import nn

from linodenet.mappings.linear import LinearContraction
from linodenet.mappings.scalar_contractions import NonExpansiveMapping

from .base import TransformSequence
from .residual_contraction import (
    DEFAULT_REZERO_SCALAR_MAP,
    ResidualBottleneck,
)


class BottleneckFlow(TransformSequence[ResidualBottleneck]):
    r"""Invertible residual flow built from low-rank bottleneck blocks."""

    input_size: Final[int]
    r"""CONST: Input and output dimensionality."""
    num_blocks: Final[int]
    r"""CONST: Number of residual blocks."""
    layers_per_block: Final[int]
    r"""CONST: Number of bottleneck layers inside each block."""
    hidden_size: Final[int]
    r"""CONST: Bottleneck dimensionality used inside each block."""
    use_rezero: Final[bool]
    r"""CONST: Whether to wrap blocks in ``ReZero`` gating."""

    def __init__(
        self,
        input_size: int,
        *,
        num_blocks: int,
        hidden_size: int,
        layers_per_block: int = 1,
        bottleneck_activation: str | nn.Module = "elu",
        use_rezero: bool = True,
        scalar_map: nn.Module | str = DEFAULT_REZERO_SCALAR_MAP,
        use_bias: bool = True,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        self.input_size = input_size
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        self.hidden_size = hidden_size
        self.use_rezero = use_rezero
        if scalar_map is not DEFAULT_REZERO_SCALAR_MAP and not self.use_rezero:
            warnings.warn(
                "Ignoring scalar_map because use_rezero=False.",
                stacklevel=2,
            )
            scalar_map = DEFAULT_REZERO_SCALAR_MAP

        blocks = [
            self._make_block(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                layers_per_block=self.layers_per_block,
                bottleneck_activation=bottleneck_activation,
                use_rezero=self.use_rezero,
                scalar_map=scalar_map,
                use_bias=use_bias,
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
        hidden_size: int,
        layers_per_block: int,
        bottleneck_activation: str | nn.Module,
        use_rezero: bool,
        scalar_map: nn.Module | str,
        use_bias: bool,
        maxiter: int,
        atol: float,
        rtol: float,
    ) -> ResidualBottleneck:
        if layers_per_block < 1:
            raise ValueError("layers_per_block must be at least 1")

        layers = [
            module
            for _ in range(layers_per_block)
            for module in (
                NonExpansiveMapping.new(bottleneck_activation),
                LinearContraction(hidden_size, hidden_size, bias=use_bias),
            )
        ]
        bottleneck = nn.Sequential(*layers)

        return ResidualBottleneck(
            input_size=input_size,
            hidden_size=hidden_size,
            bottleneck=bottleneck,
            gate="rezero" if use_rezero else "identity",
            scalar_map=scalar_map,
            use_bias=use_bias,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
        )
