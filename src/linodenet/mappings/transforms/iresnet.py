r"""Simple i-ResNet built from residual contraction blocks."""

__all__ = ["IResNet"]

import warnings
from typing import Final

from torch import nn

from linodenet.mappings.linear import LinearContraction
from linodenet.nn.activations import Activations

from .base import TransformSequence
from .residual_contraction import DEFAULT_REZERO_SCALAR_MAP, ResidualContraction


class IResNet(TransformSequence[ResidualContraction]):
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
        scalar_map: nn.Module | str = DEFAULT_REZERO_SCALAR_MAP,
        maxiter: int = 256,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        trace_estimator: str = "hutch",
        trace_matvecs: int = 3,
        logdet_series_terms: int = 8,
        trace_probe_sampler: str = "sphere",
        trace_mode: str = "reverse",
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
        if scalar_map is not DEFAULT_REZERO_SCALAR_MAP and not self.use_rezero:
            warnings.warn(
                "Ignoring scalar_map because use_rezero=False.",
                stacklevel=2,
            )
            scalar_map = DEFAULT_REZERO_SCALAR_MAP

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
                trace_estimator=trace_estimator,
                trace_matvecs=trace_matvecs,
                logdet_series_terms=logdet_series_terms,
                trace_probe_sampler=trace_probe_sampler,
                trace_mode=trace_mode,
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
        scalar_map: nn.Module | str,
        maxiter: int,
        atol: float,
        rtol: float,
        trace_estimator: str,
        trace_matvecs: int,
        logdet_series_terms: int,
        trace_probe_sampler: str,
        trace_mode: str,
    ) -> ResidualContraction:
        layers: list[nn.Module] = []
        act = Activations.new(activation)
        assert isinstance(act, nn.Module)
        if layers_per_block < 1:
            raise ValueError("layers_per_block must be at least 1")
        if layers_per_block == 1:
            layers.extend([act, LinearContraction(input_size, input_size)])
        else:
            layers.extend([act, LinearContraction(input_size, latent_size)])
            layers.extend(
                module
                for _ in range(layers_per_block - 2)
                for module in (act, LinearContraction(latent_size, latent_size))
            )
            layers.extend([act, LinearContraction(latent_size, input_size)])

        contraction = nn.Sequential(*layers)
        return ResidualContraction(
            contraction,
            gate="rezero" if use_rezero else "identity",
            scalar_map=scalar_map,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
            trace_estimator=trace_estimator,
            trace_matvecs=trace_matvecs,
            logdet_series_terms=logdet_series_terms,
            trace_probe_sampler=trace_probe_sampler,
            trace_mode=trace_mode,
        )
