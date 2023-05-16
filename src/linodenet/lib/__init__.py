"""C++ Extensions used by LinODEnet."""

__all__ = [
    "singular_triplet",
    "spectral_norm",
    "singular_triplet_native",
    "spectral_norm_native",
]

from linodenet.lib.custom_operators import (
    singular_triplet,
    singular_triplet_native,
    spectral_norm,
    spectral_norm_native,
)
