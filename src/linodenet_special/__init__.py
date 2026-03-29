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
    "bimodal_to_gaussian_value_and_grad",
    "gaussian_to_bimodal_value_and_grad",
    "mixture_to_gaussian_value_and_grad",
]

from . import fallbacks, interfaces, kernels, linalg
from .fallbacks import (
    bimodal_to_gaussian_value_and_grad,
    fixpoint_solve,
    gaussian_to_bimodal_value_and_grad,
    inverse_softplus,
    matrix_log,
    matrix_sqrt,
    mixture_to_gaussian_value_and_grad,
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
