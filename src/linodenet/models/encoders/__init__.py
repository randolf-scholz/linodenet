r"""Encoder Models."""

__all__ = [
    # Types
    "Encoder",
    # Meta-Objects
    "ENCODERS",
    # Classes
    "iResNet",
    "iResNetBlock",
    "ResNetBlock",
    "ResNet",
    "FT_Transformer",
    "Transformer",
    "LinearContraction",
    "SpectralNorm",
    # Functions
    "spectral_norm",
]

from typing import Final

from torch import nn

from linodenet.models.encoders.ft_transformer import FT_Transformer
from linodenet.models.encoders.iresnet import (
    LinearContraction,
    SpectralNorm,
    iResNet,
    iResNetBlock,
    spectral_norm,
)
from linodenet.models.encoders.resnet import ResNet, ResNetBlock
from linodenet.models.encoders.transformer import Transformer

Encoder = nn.Module
r"""Type hint for Encoders."""

ENCODERS: Final[dict[str, type[Encoder]]] = {
    "iResNet": iResNet,
    "ResNet": ResNet,
    "FT_Transformer": FT_Transformer,
    "Transformer": Transformer,
}
r"""Dictionary containing all available encoders."""
