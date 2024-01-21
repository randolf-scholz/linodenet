r"""Models of the LinODE-Net package."""

__all__ = [
    "LSSM",
    # Constants
    "MODELS",
    "LatentStateSpaceModel",
    "LinODE",
    "LinODECell",
    "LinODEnet",
    "LinearContraction",
    # ABCs & Protocols
    "Model",
    "ModelABC",
    # Classes
    "ResNet",
    "ResNetBlock",
    "SpectralNorm",
    "iResNet",
    "iResNetBlock",
    # Sub-Packages
    "embeddings",
    "encoders",
    "filters",
    # Functions
    "spectral_norm",
    "system",
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
from linodenet.models.lssm import LatentStateSpaceModel, LatentStateSpaceModel as LSSM
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
