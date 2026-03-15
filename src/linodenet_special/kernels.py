r"""Available kernels exposed as attributes."""

__all__ = [
    # CONSTANTS
    "KERNELS",
    "COMPILED",
    "FALLBACKS",
    # Classes
    "Kernels",
    # functions
    "hard_bend",
    "ndtri_exp",
    "singular_triplet",
    "spectral_norm",
    "bimodal_to_gaussian",
    "gaussian_to_bimodal",
    "gaussian_to_mixture",
    "mixture_to_gaussian",
]

import logging
from dataclasses import dataclass
from typing import Any, Final

from linodenet_special import compiled, fallbacks
from linodenet_special.compiled import COMPILED as COMPILED_FN
from linodenet_special.interfaces import (
    BimodalToGaussian,
    GaussianToBimodal,
    GaussianToMixture,
    HardBend,
    KnownFunctions,
    MixtureToGaussian,
    NdtriExp,
    SingularTriplet,
    SpectralNorm,
)

logging.basicConfig(level=logging.WARNING)
__logger__ = logging.getLogger(__package__)


@dataclass(frozen=True)
class Kernels:
    r"""The selected kernels exposed as attributes."""

    singular_triplet: SingularTriplet
    spectral_norm: SpectralNorm
    ndtri_exp: NdtriExp
    hard_bend: HardBend
    bimodal_to_gaussian: BimodalToGaussian
    gaussian_to_bimodal: GaussianToBimodal
    gaussian_to_mixture: GaussianToMixture
    mixture_to_gaussian: MixtureToGaussian


def _select_fns() -> Kernels:
    r"""Select compiled kernels when available and fall back otherwise."""
    impls: dict[str, Any] = {}
    missing: set[str] = set()
    for name in KnownFunctions.__required_keys__:
        if (compiled_fn := COMPILED_FN.get(name)) is not None:
            impls[name] = compiled_fn
        elif (fallback_fn := FALLBACKS.get(name)) is not None:
            impls[name] = fallback_fn
            missing.add(name)
        else:
            raise RuntimeError(
                f"Neither compiled nor fallback implementation found for kernel '{name}'!"
            )

    if missing:
        __logger__.warning(
            "Missing compiled versions of the following kernels:"
            f"\n\t- {'\n\t- '.join(missing)}"
            "\nUsing pure python fallbacks for these functions."
        )

    return Kernels(**impls)


FALLBACKS: Final[KnownFunctions] = {
    "ndtri_exp"           : fallbacks.ndtri_exp,
    "singular_triplet"    : fallbacks.singular_triplet,
    "spectral_norm"       : fallbacks.spectral_norm,
    "hard_bend"           : fallbacks.hard_bend,
    "bimodal_to_gaussian" : fallbacks.bimodal_to_gaussian,
    "gaussian_to_bimodal" : fallbacks.gaussian_to_bimodal,
    "gaussian_to_mixture" : fallbacks.gaussian_to_mixture,
    "mixture_to_gaussian" : fallbacks.mixture_to_gaussian,
}  # fmt: skip


COMPILED: Final[KnownFunctions] = {
    "ndtri_exp": compiled.ndtri_exp,
    "singular_triplet": compiled.singular_triplet,
    "spectral_norm": compiled.spectral_norm,
    "hard_bend": compiled.hard_bend,
    "bimodal_to_gaussian": compiled.bimodal_to_gaussian,
    "gaussian_to_bimodal": compiled.gaussian_to_bimodal,
    "gaussian_to_mixture": compiled.gaussian_to_mixture,
    "mixture_to_gaussian": compiled.mixture_to_gaussian,
}


KERNELS: Final[Kernels] = _select_fns()

# fmt: off
hard_bend:           HardBend          = KERNELS.hard_bend
ndtri_exp:           NdtriExp          = KERNELS.ndtri_exp
singular_triplet:    SingularTriplet   = KERNELS.singular_triplet
spectral_norm:       SpectralNorm      = KERNELS.spectral_norm
bimodal_to_gaussian: BimodalToGaussian = KERNELS.bimodal_to_gaussian
gaussian_to_bimodal: GaussianToBimodal = KERNELS.gaussian_to_bimodal
gaussian_to_mixture: GaussianToMixture = KERNELS.gaussian_to_mixture
mixture_to_gaussian: MixtureToGaussian = KERNELS.mixture_to_gaussian
# fmt: on
