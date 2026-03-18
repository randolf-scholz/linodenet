r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    # Sub-Packages
    "bijections",
    "distributions",
    "embeddings",
    "filters",
    "flows",
    "forecasting",
    "imputation",
    "initializations",
    "mappings",
    "nn",
    "parametrizations",
    "projections",
    "regularizations",
    "registry",
    "signatures",
    "special",
    "surjections",
    "testing",
    "transforms",
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
    constants,
    distributions,
    domains,
    filters,
    flows,
    forecasting,
    imputation,
    initializations,
    mappings,
    nn,
    parametrizations,
    registry,
    regularizations,
    testing,
    types,
    utils,
)
from linodenet.mappings import (
    bijections,
    embeddings,
    projections,
    surjections,
    transforms,
)
