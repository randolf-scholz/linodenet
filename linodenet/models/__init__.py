r"""Models of the LinODE-Net package."""

import logging
from typing import Final

from torch.nn import Module

from linodenet.models.iResNet import LinearContraction, iResNet, iResNetBlock
from linodenet.models.LinODEnet import (
    ConcatEmbedding,
    ConcatProjection,
    LinODE,
    LinODECell,
    LinODEnet,
)

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = ["MODELS", "Model"] + [  # Classes
    "LinearContraction",
    "iResNetBlock",
    "iResNet",
    "LinODECell",
    "LinODE",
    "LinODEnet",
    "ConcatProjection",
    "ConcatEmbedding",
]

Model = Module
r"""Type hint for models."""

MODELS: Final[dict[str, type[Model]]] = {
    "LinearContraction": LinearContraction,
    "iResNetBlock": iResNetBlock,
    "iResNet": iResNet,
    "LinODECell": LinODECell,
    "LinODE": LinODE,
    "LinODEnet": LinODEnet,
    "ConcatProjection": ConcatProjection,
    "ConcatEmbedding": ConcatEmbedding,
}
r"""Dictionary containing all available models."""
