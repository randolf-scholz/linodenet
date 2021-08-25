r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

import logging
from pathlib import Path
from typing import Final

from linodenet import init, models

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["__version__", "init", "models"]

with open(Path(__file__).parent.joinpath("VERSION"), "r", encoding="utf-8") as file:
    __version__ = file.read()
