r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    "CONFIG",
    # Functions
    "info",
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

# single-source version
try:
    __version__ = metadata.version(__package__ or __name__)
    r"""The version number of the `tsdm` package."""
except metadata.PackageNotFoundError:
    __version__ = "unknown"
    r"""The version number of the `tsdm` package."""
finally:
    del metadata
