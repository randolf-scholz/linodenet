r"""C++ Extensions used by LinODEnet."""

__all__ = [
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
    "spectral_norm_native",
    "spectral_norm_riemann",
    # Linalg
    "pad",
    "scaled_norm",
    "geometric_mean",
    # Special
    "ndtri_exp",
    "ndtri_exp_naive",
]

from linodenet.lib._liblinodenet import (
    SingularTriplet,
    SpectralNorm,
    singular_triplet,
    singular_triplet_debug,
    singular_triplet_native,
    singular_triplet_riemann,
    spectral_norm,
    spectral_norm_debug,
    spectral_norm_native,
    spectral_norm_riemann,
)
from linodenet.lib.linalg import (
    geometric_mean,
    pad,
    scaled_norm,
)
from linodenet.lib.ndtri_exp import ndtri_exp, ndtri_exp_naive
