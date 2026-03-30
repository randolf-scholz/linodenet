r"""Available kernels exposed as attributes."""

__all__ = [
    # CONSTANTS
    "KERNELS",
    "FALLBACKS",
    "COMPILED",
    # Classes
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
from typing import Any, Final

from . import compiled as CPP, fallbacks as PY
from .interfaces import Kernels, KnownFunctions

logging.basicConfig(level=logging.WARNING)
__logger__ = logging.getLogger(__package__)


def _select_fns() -> Kernels:
    r"""Select compiled kernels when available and fall back otherwise."""
    impls: dict[str, Any] = {}
    compiled_missing: set[str] = set()
    for name in KnownFunctions.__required_keys__:
        if name in CPP.AVAILABLE_KERNELS:
            impls[name] = getattr(COMPILED, name)
        elif name in PY.__all__:
            impls[name] = getattr(FALLBACKS, name)
            compiled_missing.add(name)
        else:
            raise RuntimeError(
                f"Neither compiled nor fallback implementation found for kernel '{name}'!"
            )

    if compiled_missing:
        __logger__.warning(
            "Missing compiled versions of the following kernels:"
            f"\n\t- {'\n\t- '.join(compiled_missing)}"
            "\nUsing pure python fallbacks for these functions."
        )

    return Kernels(**impls)


FALLBACKS: Final[Kernels] = Kernels(
    ndtri_exp=PY.ndtri_exp,
    singular_triplet=PY.singular_triplet,
    spectral_norm=PY.spectral_norm,
    hard_bend=PY.hard_bend,
    bimodal_to_gaussian=PY.bimodal_to_gaussian,
    bimodal_to_gaussian_value_and_grad=PY.bimodal_to_gaussian_value_and_grad,
    gaussian_to_bimodal=PY.gaussian_to_bimodal,
    gaussian_to_bimodal_value_and_grad=PY.gaussian_to_bimodal_value_and_grad,
    gaussian_to_mixture=PY.gaussian_to_mixture,
    gaussian_to_mixture_value_and_grad=PY.gaussian_to_mixture_value_and_grad,
    mixture_to_gaussian=PY.mixture_to_gaussian,
    mixture_to_gaussian_value_and_grad=PY.mixture_to_gaussian_value_and_grad,
)

COMPILED: Final[Kernels] = Kernels(
    ndtri_exp=CPP.ndtri_exp,
    singular_triplet=CPP.singular_triplet,
    spectral_norm=CPP.spectral_norm,
    hard_bend=CPP.hard_bend,
    bimodal_to_gaussian=CPP.bimodal_to_gaussian,
    bimodal_to_gaussian_value_and_grad=CPP.bimodal_to_gaussian_value_and_grad,
    gaussian_to_bimodal=CPP.gaussian_to_bimodal,
    gaussian_to_bimodal_value_and_grad=CPP.gaussian_to_bimodal_value_and_grad,
    gaussian_to_mixture=CPP.gaussian_to_mixture,
    gaussian_to_mixture_value_and_grad=CPP.gaussian_to_mixture_value_and_grad,
    mixture_to_gaussian=CPP.mixture_to_gaussian,
    mixture_to_gaussian_value_and_grad=CPP.mixture_to_gaussian_value_and_grad,
)

KERNELS: Final[Kernels] = _select_fns()
# fmt: off
hard_bend: Final            = KERNELS.hard_bend
ndtri_exp: Final            = KERNELS.ndtri_exp
singular_triplet: Final     = KERNELS.singular_triplet
spectral_norm: Final        = KERNELS.spectral_norm
bimodal_to_gaussian: Final  = KERNELS.bimodal_to_gaussian
gaussian_to_bimodal: Final  = KERNELS.gaussian_to_bimodal
gaussian_to_mixture: Final  = KERNELS.gaussian_to_mixture
mixture_to_gaussian: Final  = KERNELS.mixture_to_gaussian
bimodal_to_gaussian_value_and_grad: Final = KERNELS.bimodal_to_gaussian_value_and_grad
gaussian_to_bimodal_value_and_grad: Final = KERNELS.gaussian_to_bimodal_value_and_grad
gaussian_to_mixture_value_and_grad: Final = KERNELS.gaussian_to_mixture_value_and_grad
mixture_to_gaussian_value_and_grad: Final = KERNELS.mixture_to_gaussian_value_and_grad
# fmt: on
