r"""Models of the LinODE-Net package."""

__all__ = [
    # sub-packages
    "embeddings",
    "encoders",
    "filters",
    "forecasting",
    "system",
    # Constants
    "MODULES",
    # Classes
    "LatentStateSpaceModel",
    "LinODE",
    "LinODECell",
    "LinODEnet",
    "LinearContraction",
    "ResNet",
    "ResNetBlock",
    "SpectralNorm",
    "iResNet",
    "iResNetBlock",
    # Functions
    "spectral_norm",
]

from linodenet.models import embeddings, encoders, filters, forecasting, system
from linodenet.models.encoders import (
    LinearContraction,
    ResNet,
    ResNetBlock,
    SpectralNorm,
    iResNet,
    iResNetBlock,
    spectral_norm,
)
from linodenet.models.forecasting import LatentStateSpaceModel, LinODEnet
from linodenet.models.system import LinODE, LinODECell

MODULES: dict[str, type] = {
    "LinODE"            : LinODE,
    "LinODECell"        : LinODECell,
    "LinODEnet"         : LinODEnet,
    "LinearContraction" : LinearContraction,
    "iResNet"           : iResNet,
    "iResNetBlock"      : iResNetBlock,
}  # fmt: skip
r"""Dictionary containing all available models."""
