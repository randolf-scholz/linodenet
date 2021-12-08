r"""Encoder Models."""

__all__ = [
    # Types
    "Encoder",
    # Meta-Objects
    "ENCODERS",
    # Classes
    "iResNet",
    "ResNet",
    "Transformer",
]

import logging
from typing import Final

from torch import nn

from linodenet.models.encoders.ft_transformer import Transformer
from linodenet.models.encoders.iresnet import iResNet
from linodenet.models.encoders.resnet import ResNet

__logger__ = logging.getLogger(__name__)

Encoder = nn.Module
r"""Type hint for Encoders."""

ENCODERS: Final[dict[str, type[Encoder]]] = {
    "iResNet": iResNet,
    "ResNet": ResNet,
    "Transformer": Transformer,
}
r"""Dictionary containing all available encoders."""
