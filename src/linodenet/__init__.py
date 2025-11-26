r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    # Sub-Packages
    "distributions",
    "parametrize",
    "projections",
    "regularizations",
    "testing",
    "initializations",
    "lib",
    "nn",
    # Sub-Modules
    "config",
    "constants",
    "context",
    "domains",
    "linalg",
    "types",
    "utils",
]


if __name__ == "__main__":
    raise RuntimeError("This library is not meant to be run directly.")


from importlib import metadata

try:  # single-source version
    __version__ = metadata.version(__package__ or __name__)
    r"""The version number of the `tsdm` package."""
except metadata.PackageNotFoundError:
    __version__ = "unknown"
    r"""The version number of the `tsdm` package."""
finally:
    del metadata


from linodenet import (
    config,
    constants,
    context,
    distributions,
    domains,
    initializations,
    lib,
    linalg,
    nn,
    parametrize,
    projections,
    regularizations,
    testing,
    types,
    utils,
)
