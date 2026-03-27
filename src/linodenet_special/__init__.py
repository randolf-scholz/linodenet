r"""C++ Extensions used by LinODEnet."""
# ruff: noqa: F403

__all__ = [
    # submodules
    "kernels",
    "fallbacks",
    "interfaces",
    # Special
    "fixpoint_solve",
    "inverse_softplus",
    "matrix_log",
    "matrix_sqrt",
    "singular_triplet_native",
    "spectral_norm_native",
]

from . import fallbacks, interfaces, kernels, linalg
from .fallbacks import (
    fixpoint_solve,
    inverse_softplus,
    matrix_log,
    matrix_sqrt,
    singular_triplet_native,
    spectral_norm_native,
)
from .interfaces import *
from .kernels import *
from .linalg import *

__all__ += kernels.__all__
__all__ += interfaces.__all__
__all__ += fallbacks.__all__
__all__ += linalg.__all__
