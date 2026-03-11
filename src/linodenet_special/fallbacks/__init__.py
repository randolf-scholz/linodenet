r"""Pure python implementations of special functions used by LinODEnet.

Used as fallback when C++ extensions are not available.
"""

__all__ = [
    "ndtri_exp_fallback",
    "ndtri_exp_naive",
    "bimodal_to_gaussian",
    "gaussian_to_bimodal",
    "gaussian_to_mixture",
    "gaussian_to_twin",
    "mixture_to_gaussian",
    "twin_to_gaussian",
    "spectral_norm",
]

from linodenet_special.fallbacks.ndtri_exp import ndtri_exp_fallback, ndtri_exp_naive
from linodenet_special.fallbacks.spectral_norm import spectral_norm
from linodenet_special.fallbacks.transport import (
    bimodal_to_gaussian,
    gaussian_to_bimodal,
    gaussian_to_mixture,
    gaussian_to_twin,
    mixture_to_gaussian,
    twin_to_gaussian,
)
