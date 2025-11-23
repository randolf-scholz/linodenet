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
    "ResNet",
    "ResNetBlock",
    "iResNet",
    "iResNetBlock",
]

from linodenet import embeddings, forecasting
from linodenet.components import encoders, filters, system
from linodenet.components.encoders import (
    ResNet,
    ResNetBlock,
    iResNet,
    iResNetBlock,
)
from linodenet.components.system import LinODE, LinODECell
from linodenet.layers.linear_contraction import LinearContraction

MODULES: dict[str, type] = {
    "LinODE"            : LinODE,
    "LinODECell"        : LinODECell,
    "LinearContraction" : LinearContraction,
    "iResNet"           : iResNet,
    "iResNetBlock"      : iResNetBlock,
}  # fmt: skip
r"""Dictionary containing all available models."""
