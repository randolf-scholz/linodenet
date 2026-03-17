r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    # Sub-Packages
    "bijections",
    "distributions",
    "filters",
    "flows",
    "forecasting",
    "imputation",
    "initializations",
    "special",
    "parametrize",
    "projections",
    "regularizations",
    "signatures",
    "testing",
    # Sub-Modules
    "constants",
    "domains",
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

import linodenet_special as special
import signatures
from linodenet import (
    bijections,
    constants,
    distributions,
    domains,
    filters,
    flows,
    forecasting,
    imputation,
    initializations,
    parametrize,
    regularizations,
    testing,
    types,
    utils,
)
from linodenet.mappings import projections
