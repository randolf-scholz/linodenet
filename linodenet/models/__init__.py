r"""Models of the LinODE-Net package."""

__all__ = [
    # Type Hint
    "Model",
    # Constants
    "MODELS",
    # Classes
    "SpectralNorm",
    "LinearContraction",
    "iResNetBlock",
    "iResNet",
    "LinODECell",
    "LinODE",
    "LinODEnet",
    # Functions
    "spectral_norm",
]

import logging
from typing import Final

from torch.nn import Module

from linodenet.models._iresnet import (
    LinearContraction,
    SpectralNorm,
    iResNet,
    iResNetBlock,
    spectral_norm,
)
from linodenet.models._linodenet import LinODE, LinODECell, LinODEnet

LOGGER = logging.getLogger(__name__)

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
