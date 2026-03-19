r"""Pure python implementations of special functions used by LinODEnet.

Used as fallback when C++ extensions are not available.
"""

__all__ = [
    # functions
    "bimodal_to_gaussian",
    "gaussian_to_bimodal",
    "gaussian_to_mixture",
    "hard_bend",
    "hard_contract",
    "hard_expand",
    "inverse_softplus",
    "matrix_log",
    "matrix_sqrt",
    "mixture_to_gaussian",
    "ndtri_exp",
    "ndtri_exp_naive",
    "singular_triplet",
    "singular_triplet_native",
    "spectral_norm",
    "spectral_norm_native",
]

from .gaussian_transport import (
    bimodal_to_gaussian,
    gaussian_to_bimodal,
    gaussian_to_mixture,
    mixture_to_gaussian,
)
from .hard_bend import hard_bend, hard_contract, hard_expand
from .inverse_softplus import inverse_softplus
from .matrix_functions import matrix_log, matrix_sqrt
from .ndtri_exp import ndtri_exp, ndtri_exp_naive
from .singular_triplet import singular_triplet, singular_triplet_native
from .spectral_norm import spectral_norm, spectral_norm_native
