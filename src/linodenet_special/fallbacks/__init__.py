r"""Pure python implementations of special functions used by LinODEnet.

Used as fallback when C++ extensions are not available.
"""

__all__ = [
    "FALLBACKS",
    "HardBend",
    "NdtriExp",
    "SingularTriplet",
    "SpectralNorm",
    # functions
    "ndtri_exp",
    "ndtri_exp_naive",
    "gaussian_to_mixture",
    "gaussian_to_bimodal",
    "mixture_to_gaussian",
    "bimodal_to_gaussian",
    "singular_triplet",
    "singular_triplet_native",
    "spectral_norm",
    "hard_bend",
    "hard_contract",
    "hard_expand",
]

from linodenet_special.fallbacks.gaussian_transport import (
    bimodal_to_gaussian,
    gaussian_to_bimodal,
    gaussian_to_mixture,
    mixture_to_gaussian,
)
from linodenet_special.fallbacks.hard_bend import (
    HardBend,
    hard_bend,
    hard_contract,
    hard_expand,
)
from linodenet_special.fallbacks.ndtri_exp import NdtriExp, ndtri_exp, ndtri_exp_naive
from linodenet_special.fallbacks.singular_triplet import (
    SingularTriplet,
    singular_triplet,
    singular_triplet_native,
)
from linodenet_special.fallbacks.spectral_norm import SpectralNorm, spectral_norm

FALLBACKS = {
    "ndtri_exp"           : ndtri_exp,
    "singular_triplet"    : singular_triplet,
    "spectral_norm"       : spectral_norm,
    "hard_bend"           : hard_bend,
    "hard_contract"       : hard_contract,
    "hard_expand"         : hard_expand,
    "bimodal_to_gaussian" : bimodal_to_gaussian,
    "gaussian_to_bimodal" : gaussian_to_bimodal,
    "gaussian_to_mixture" : gaussian_to_mixture,
    "mixture_to_gaussian" : mixture_to_gaussian,
}  # fmt: skip
r"""Pure python implementations of special functions used by LinODEnet."""
