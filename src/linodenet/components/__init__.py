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
    "LinODE",
    "LinODECell",
    "LinearContraction",
    "ResNet",
    "ResNetBlock",
    "SpectralNorm",
    "iResNet",
    "iResNetBlock",
]

from linodenet import embeddings, forecasting
from linodenet.components import encoders, filters, system
from linodenet.components.encoders import (
    LinearContraction,
    ResNet,
    ResNetBlock,
    SpectralNorm,
    iResNet,
    iResNetBlock,
)
from linodenet.components.system import LinODE, LinODECell

MODULES: dict[str, type] = {
    "LinODE"            : LinODE,
    "LinODECell"        : LinODECell,
    "LinearContraction" : LinearContraction,
    "iResNet"           : iResNet,
    "iResNetBlock"      : iResNetBlock,
}  # fmt: skip
r"""Dictionary containing all available models."""
