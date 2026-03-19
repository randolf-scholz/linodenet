r"""C++ Extensions used by LinODEnet."""
# ruff: noqa: F403

__all__ = [
    # submodules
    "kernels",
    "fallbacks",
    "interfaces",
    # Special
    "thomson_initialization",
    "spectral_norm_native",
    "singular_triplet_native",
    "inverse_softplus",
    "matrix_sqrt",
    "matrix_log",
]
from linodenet_special import fallbacks, interfaces, kernels, linalg
from linodenet_special.fallbacks import (
    inverse_softplus,
    matrix_log,
    matrix_sqrt,
    singular_triplet_native,
    spectral_norm_native,
)
from linodenet_special.interfaces import *
from linodenet_special.kernels import *
from linodenet_special.linalg import *
from linodenet_special.thomson_initialization import thomson_initialization

__all__ += kernels.__all__
__all__ += interfaces.__all__
__all__ += fallbacks.__all__
__all__ += linalg.__all__
