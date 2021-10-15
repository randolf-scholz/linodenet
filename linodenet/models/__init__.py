r"""Models of the LinODE-Net package."""

__all__ = [
    # Type Hint
    "Model",
    # Constants
    "MODELS",
    # Classes
    "LinearContraction",
    "iResNetBlock",
    "iResNet",
    "LinODECell",
    "LinODE",
    "LinODEnet",
]

import logging
from typing import Final

from torch.nn import Module

from linodenet.models.iresnet import LinearContraction, iResNet, iResNetBlock
from linodenet.models.linodenet import LinODE, LinODECell, LinODEnet

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


# print(globals().keys())
# for key in __all__:
#     print(f">>>{key}<<<", globals()[key])
#     if isinstance(globals()[key], type) or callable(globals()[key]):
#         globals()[key].__module__ = __name__
