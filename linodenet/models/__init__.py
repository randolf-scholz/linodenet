r"""Models of the LinODE-Net package."""

__all__ = [
    # Sub-Packages
    "filters",
    "encoders",
    "embeddings",
    # Type Hint
    "Model",
    # Constants
    "MODELS",
    # Classes
    "SpectralNorm",
    "LinearContraction",
    "iResNetBlock",
    "iResNet",
    "ResNet",
    "ResNetBlock",
    "LinODE",
    "LinODEnet",
    "LinODECell",
    # Functions
    "spectral_norm",
]

import logging
from typing import Final

from torch.nn import Module

from linodenet.models import embeddings, encoders, filters
from linodenet.models._linodenet import LinODE, LinODEnet
from linodenet.models.encoders import (
    LinearContraction,
    ResNet,
    ResNetBlock,
    SpectralNorm,
    iResNet,
    iResNetBlock,
    spectral_norm,
)
from linodenet.models.system import LinODECell

__logger__ = logging.getLogger(__name__)

Model = Module
r"""Type hint for models."""

MODELS: Final[dict[str, type[Model]]] = {
    "LinearContraction": LinearContraction,
    "iResNetBlock": iResNetBlock,
    "iResNet": iResNet,
    "LinODECell": LinODECell,
    "LinODE": LinODE,
    "LinODEnet": LinODEnet,
}
r"""Dictionary containing all available models."""
