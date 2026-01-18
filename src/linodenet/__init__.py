r"""Linear Ordinary Differential Equation Recurrent Neural Network."""

__all__ = [
    # Constants
    "__version__",
    # Sub-Packages
    "activations",
    "bijections",
    "distributions",
    "embeddings",
    "encoders",
    "filters",
    "forecasting",
    "imputation",
    "initializations",
    "lib",
    "parametrize",
    "projections",
    "regularizations",
    "signatures",
    "system",
    "testing",
    # Sub-Modules
    "containers",
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
    activations,
    bijections,
    config,
    constants,
    containers,
    context,
    distributions,
    domains,
    embeddings,
    encoders,
    filters,
    forecasting,
    imputation,
    initializations,
    lib,
    linalg,
    parametrize,
    projections,
    regularizations,
    signatures,
    system,
    testing,
    types,
    utils,
)
