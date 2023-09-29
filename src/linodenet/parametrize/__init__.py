"""Parametrizations for Torch."""

__all__ = [
    # Classes
    "ParametrizationProto",
    "Parametrization",
    "SimpleParametrization",
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

from linodenet.parametrize.base import (
    Parametrization,
    ParametrizationProto,
    SimpleParametrization,
    get_parametrizations,
    register_parametrization,
    reset_all_caches,
)
from linodenet.parametrize.parametrizations import ReZero, SpectralNormalization

PARAMETRIZATIONS = {
    "SpectralNormalization": SpectralNormalization,
    "ReZero": ReZero,
}
