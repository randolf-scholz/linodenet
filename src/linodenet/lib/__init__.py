"""C++ Extensions used by LinODEnet."""

__all__ = [
    # Functions
    "pad",
    "singular_triplet",
    "singular_triplet_debug",
    "singular_triplet_native",
    "singular_triplet_riemann",
    "spectral_norm",
    "spectral_norm_debug",
    "spectral_norm_native",
    "spectral_norm_riemann",
]

from linodenet.lib._liblinodenet import (
    singular_triplet,
    singular_triplet_debug,
    singular_triplet_native,
    singular_triplet_riemann,
    spectral_norm,
    spectral_norm_debug,
    spectral_norm_native,
    spectral_norm_riemann,
)
from linodenet.lib._utils import pad
