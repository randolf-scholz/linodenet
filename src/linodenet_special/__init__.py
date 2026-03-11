r"""C++ Extensions used by LinODEnet."""

__all__ = [
    "KnownFunctions",
    # Protocols
    "SpectralNorm",
    "SingularTriplet",
    # Functions
    "singular_triplet",
    "singular_triplet_debug",
    "singular_triplet_native",
    "singular_triplet_riemann",
    "spectral_norm",
    "spectral_norm_debug",
    "spectral_norm_riemann",
    "spectral_norm_native",
    # Linalg
    "pad",
    "scaled_norm",
    "geometric_mean",
    # Special
    "ndtri_exp",
    "thomson_initialization",
]

from linodenet_special.core import (
    KnownFunctions,
    ndtri_exp,
    singular_triplet,
    singular_triplet_debug,
    singular_triplet_riemann,
    spectral_norm,
    spectral_norm_debug,
    spectral_norm_riemann,
)
from linodenet_special.fallbacks.singular_triplet import (
    SingularTriplet,
    singular_triplet_native,
)
from linodenet_special.fallbacks.spectral_norm import SpectralNorm, spectral_norm_native
from linodenet_special.linalg import (
    geometric_mean,
    pad,
    scaled_norm,
)
from linodenet_special.thomson_initialization import thomson_initialization
