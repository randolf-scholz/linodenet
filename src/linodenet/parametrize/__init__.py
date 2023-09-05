"""Parametrizations for Torch."""

__all__ = [
    # Classes
    "ParametrizationProto",
    "Parametrization",
    "Parametrize",
    # functions
    "register_parametrization",
    "get_parametrizations",
    "reset_all_caches",
    # Parametrizations
    "SpectralNormalization",
    "ReZero",
    # Constants
    "PARAMETRIZATIONS",
]

from linodenet.parametrize._parametrize import (
    Parametrization,
    ParametrizationProto,
    Parametrize,
    get_parametrizations,
    register_parametrization,
    reset_all_caches,
)
from linodenet.parametrize.parametrizations import ReZero, SpectralNormalization

PARAMETRIZATIONS = {
    "SpectralNormalization": SpectralNormalization,
    "ReZero": ReZero,
}
