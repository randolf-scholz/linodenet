r"""Models of the LinODE-Net package."""

__all__ = [
    # Sub-Packages
    "filters",
    "encoders",
    "embeddings",
    "system",
    # Types
    "Model",
    "ModelABC",
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
    "LatentStateSpaceModel",
    # Functions
    "spectral_norm",
    # Filters
]

from linodenet.models import embeddings, encoders, filters, system
from linodenet.models._linodenet import LinODE, LinODEnet
from linodenet.models._models import Model, ModelABC
from linodenet.models.encoders import (
    LinearContraction,
    ResNet,
    ResNetBlock,
    SpectralNorm,
    iResNet,
    iResNetBlock,
    spectral_norm,
)
from linodenet.models.lssm import LatentStateSpaceModel
from linodenet.models.system import LinODECell

MODELS: dict[str, type[Model]] = {
    "LinearContraction": LinearContraction,
    "iResNetBlock": iResNetBlock,
    "iResNet": iResNet,
    "LinODECell": LinODECell,
    "LinODE": LinODE,
    "LinODEnet": LinODEnet,
}
r"""Dictionary containing all available models."""
