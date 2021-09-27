r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = (
    "__version__",
    "config",
    "models",
    "initializations",
    "regularizations",
    "projections",
)

import logging
from pathlib import Path

from linodenet.config import GlobalConfig as config  # noqa # isort: skip
from linodenet import (  # isort: skip
    initializations,
    models,
    projections,
    regularizations,
)

LOGGER = logging.getLogger(__name__)

with open(Path(__file__).parent.joinpath("VERSION"), "r", encoding="utf8") as file:
    __version__ = file.read()
