r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

from pathlib import Path

from linodenet import init, models


with open(Path(__file__).parent.joinpath("VERSION"), "r") as file:
    __version__ = file.read()

__all__ = ["__version__", "init", "models"]
