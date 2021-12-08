r"""Encoder Models."""

__all__ = [
    # Types
    "Encoder",
    # Meta-Objects
    "ENCODERS",
    # Classes
    "iResNet",
    "ResNet",
    "FT_Transformer",
    "TransformerEncoder",
    "Transformer",
]

import logging
from typing import Final

from torch import nn

from linodenet.models.encoders.ft_transformer import FT_Transformer
from linodenet.models.encoders.iresnet import iResNet
from linodenet.models.encoders.resnet import ResNet
from linodenet.models.encoders.transformer import Transformer, TransformerEncoder

__logger__ = logging.getLogger(__name__)

Encoder = nn.Module
r"""Type hint for Encoders."""

ENCODERS: Final[dict[str, type[Encoder]]] = {
    "iResNet": iResNet,
    "ResNet": ResNet,
    "FT_Transformer": FT_Transformer,
    "TransformerEncoder": TransformerEncoder,
}
r"""Dictionary containing all available encoders."""
