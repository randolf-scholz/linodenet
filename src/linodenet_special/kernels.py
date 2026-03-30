r"""Available kernels exposed as attributes."""

__all__ = [
    # CONSTANTS
    "KERNELS",
    "FALLBACKS",
    "COMPILED",
    # Classes
    "Kernels",
    # functions
    "hard_bend",
    "ndtri_exp",
    "singular_triplet",
    "spectral_norm",
    "bimodal_to_gaussian",
    "bimodal_to_gaussian_value_and_grad",
    "gaussian_to_bimodal",
    "gaussian_to_bimodal_value_and_grad",
    "gaussian_to_mixture",
    "gaussian_to_mixture_value_and_grad",
    "mixture_to_gaussian",
    "mixture_to_gaussian_value_and_grad",
]

import logging
from dataclasses import dataclass
from typing import Any, Final

from . import compiled, fallbacks
from .compiled import WRAPPED_KERNELS
from .interfaces import (
    BimodalToGaussian,
    BimodalToGaussianValueAndGrad,
    GaussianToBimodal,
    GaussianToBimodalValueAndGrad,
    GaussianToMixture,
    GaussianToMixtureValueAndGrad,
    HardBend,
    KnownFunctions,
    MixtureToGaussian,
    MixtureToGaussianValueAndGrad,
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
    bimodal_to_gaussian_value_and_grad: BimodalToGaussianValueAndGrad
    gaussian_to_bimodal: GaussianToBimodal
    gaussian_to_bimodal_value_and_grad: GaussianToBimodalValueAndGrad
    gaussian_to_mixture: GaussianToMixture
    gaussian_to_mixture_value_and_grad: GaussianToMixtureValueAndGrad
    mixture_to_gaussian: MixtureToGaussian
    mixture_to_gaussian_value_and_grad: MixtureToGaussianValueAndGrad


def _select_fns() -> Kernels:
    r"""Select compiled kernels when available and fall back otherwise."""
    impls: dict[str, Any] = {}
    missing: set[str] = set()
    for name in KnownFunctions.__required_keys__:
        if (compiled_fn := WRAPPED_KERNELS.get(name)) is not None:
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
    "bimodal_to_gaussian_value_and_grad" : fallbacks.bimodal_to_gaussian_value_and_grad,
    "gaussian_to_bimodal" : fallbacks.gaussian_to_bimodal,
    "gaussian_to_bimodal_value_and_grad" : fallbacks.gaussian_to_bimodal_value_and_grad,
    "gaussian_to_mixture" : fallbacks.gaussian_to_mixture,
    "gaussian_to_mixture_value_and_grad" : fallbacks.gaussian_to_mixture_value_and_grad,
    "mixture_to_gaussian" : fallbacks.mixture_to_gaussian,
    "mixture_to_gaussian_value_and_grad" : fallbacks.mixture_to_gaussian_value_and_grad,
}  # fmt: skip

COMPILED: Final[KnownFunctions] = {
    "ndtri_exp"           : compiled.ndtri_exp,
    "singular_triplet"    : compiled.singular_triplet,
    "spectral_norm"       : compiled.spectral_norm,
    "hard_bend"           : compiled.hard_bend,
    "bimodal_to_gaussian" : compiled.bimodal_to_gaussian,
    "bimodal_to_gaussian_value_and_grad" : compiled.bimodal_to_gaussian_value_and_grad,
    "gaussian_to_bimodal" : compiled.gaussian_to_bimodal,
    "gaussian_to_bimodal_value_and_grad" : compiled.gaussian_to_bimodal_value_and_grad,
    "gaussian_to_mixture" : compiled.gaussian_to_mixture,
    "gaussian_to_mixture_value_and_grad" : compiled.gaussian_to_mixture_value_and_grad,
    "mixture_to_gaussian" : compiled.mixture_to_gaussian,
    "mixture_to_gaussian_value_and_grad" : compiled.mixture_to_gaussian_value_and_grad,
}  # fmt: skip

KERNELS: Final[Kernels] = _select_fns()

# fmt: off
hard_bend:           HardBend          = KERNELS.hard_bend
ndtri_exp:           NdtriExp          = KERNELS.ndtri_exp
singular_triplet:    SingularTriplet   = KERNELS.singular_triplet
spectral_norm:       SpectralNorm      = KERNELS.spectral_norm
bimodal_to_gaussian: BimodalToGaussian = KERNELS.bimodal_to_gaussian
bimodal_to_gaussian_value_and_grad: BimodalToGaussianValueAndGrad = KERNELS.bimodal_to_gaussian_value_and_grad
gaussian_to_bimodal: GaussianToBimodal = KERNELS.gaussian_to_bimodal
gaussian_to_bimodal_value_and_grad: GaussianToBimodalValueAndGrad = KERNELS.gaussian_to_bimodal_value_and_grad
gaussian_to_mixture: GaussianToMixture = KERNELS.gaussian_to_mixture
gaussian_to_mixture_value_and_grad: GaussianToMixtureValueAndGrad = KERNELS.gaussian_to_mixture_value_and_grad
mixture_to_gaussian: MixtureToGaussian = KERNELS.mixture_to_gaussian
mixture_to_gaussian_value_and_grad: MixtureToGaussianValueAndGrad = KERNELS.mixture_to_gaussian_value_and_grad
# fmt: on
