r"""Pure python implementations of special functions used by LinODEnet.

Used as fallback when C++ extensions are not available.
"""

__all__ = [
    "ndtri_exp",
    "ndtri_exp_naive",
    "bimodal_to_gaussian",
    "gaussian_to_bimodal",
    "gaussian_to_mixture",
    "gaussian_to_twin",
    "mixture_to_gaussian",
    "twin_to_gaussian",
    "singular_triplet",
    "singular_triplet_native",
    "spectral_norm",
]

from linodenet_special.fallbacks.ndtri_exp import ndtri_exp, ndtri_exp_naive
from linodenet_special.fallbacks.singular_triplet import (
    singular_triplet,
    singular_triplet_native,
)
from linodenet_special.fallbacks.spectral_norm import spectral_norm
from linodenet_special.fallbacks.transport import (
    bimodal_to_gaussian,
    gaussian_to_bimodal,
    gaussian_to_mixture,
    gaussian_to_twin,
    mixture_to_gaussian,
    twin_to_gaussian,
)
