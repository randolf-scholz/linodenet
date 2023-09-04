r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    "CONFIG",
    # Sub-Modules
    "config",
    "initializations",
    "lib",
    "models",
    "projections",
    "regularizations",
    "utils",
    "testing",
]

from importlib import metadata

from linodenet import (
    config,
    initializations,
    lib,
    models,
    projections,
    regularizations,
    testing,
    utils,
)
from linodenet.config import CONFIG

__version__ = metadata.version(__package__)
r"""The version number of the `linodenet` package."""
del metadata
