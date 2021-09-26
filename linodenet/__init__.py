r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

import logging
from pathlib import Path
from typing import Final

from linodenet import config, initializations, models, projections, regularizations

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "__version__",
    "config",
    "models",
    "initializations",
    "regularizations",
    "projections",
]

with open(Path(__file__).parent.joinpath("VERSION"), "r", encoding="utf8") as file:
    __version__ = file.read()
