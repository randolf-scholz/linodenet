r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    # Sub-Modules
    "config",
    "initializations",
    "lib",
    "models",
    "projections",
    "regularizations",
    "testing",
    "utils",
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

# single-source version
try:
    __version__ = metadata.version(__package__ or __name__)
    r"""The version number of the `tsdm` package."""
except metadata.PackageNotFoundError:
    __version__ = "unknown"
    r"""The version number of the `tsdm` package."""
finally:
    del metadata
