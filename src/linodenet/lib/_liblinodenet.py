"""Custom operators for the linodenet package."""

__all__ = [
    # module
    "LIB",
    # Protocols
    "SingularTriplet",
    "SpectralNorm",
    # Implementations
    "singular_triplet",
    "singular_triplet_debug",
    "singular_triplet_native",
    "spectral_norm",
    "spectral_norm_debug",
    "spectral_norm_native",
]

import warnings
from pathlib import Path
from typing import Optional, Protocol, cast

import torch
import torch.utils.cpp_extension
from torch import Tensor


class SpectralNorm(Protocol):
    """Protocol for spectral norm implementations."""

    def __call__(
        self,
        A: Tensor,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
        maxiter: Optional[int] = None,
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ) -> Tensor:
        ...


class SingularTriplet(Protocol):
    """Protocol for singular triplet implementations."""

    def __call__(
        self,
        A: Tensor,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
        maxiter: Optional[int] = None,
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...


def singular_triplet_native(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the singular triplet."""
    U, S, V = torch.linalg.svd(A)
    u, s, v = U[:, 0], S[0], V[0, :]
    return s, u, v


def spectral_norm_native(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Tensor:
    """Computes the spectral norm."""
    return torch.linalg.matrix_norm(A, ord=2)


lib_base_path = Path(__file__).parent
lib_path = lib_base_path / "build" / "liblinodenet.so"

_spectral_norm: SpectralNorm
_singular_triplet: SingularTriplet
_singular_triplet_debug: SingularTriplet
_spectral_norm_debug: SpectralNorm

LIB = torch.ops.liblinodenet
"""The custom library."""

if lib_path.exists():
    torch.ops.load_library(lib_path)
    _spectral_norm = cast(SpectralNorm, LIB.spectral_norm)
    _spectral_norm_debug = cast(SpectralNorm, LIB.spectral_norm_debug)
    _singular_triplet = cast(SingularTriplet, LIB.singular_triplet)
    _singular_triplet_debug = cast(SingularTriplet, LIB.singular_triplet_debug)
else:
    warnings.warn(
        "Custom binaries not found! Trying to compile them on the fly!."
        " Please compile the extension in the linodenet/lib folder via poetry build.",
        UserWarning,
        stacklevel=2,
    )

    try:
        torch.utils.cpp_extension.load(
            name="spectral_norm",
            sources=[lib_base_path / "spectral_norm.cpp"],  # type: ignore[list-item]
            is_python_module=False,
            verbose=True,
        )
        _spectral_norm = LIB.spectral_norm  # pyright: ignore

        torch.utils.cpp_extension.load(
            name="spectral_norm_debug",
            sources=[lib_base_path / "spectral_norm_debug.cpp"],  # type: ignore[list-item]
            is_python_module=False,
            verbose=True,
        )
        _spectral_norm_debug = LIB.spectral_norm_debug  # pyright: ignore

        torch.utils.cpp_extension.load(
            name="singular_triplet",
            sources=[lib_base_path / "singular_triplet.cpp"],  # type: ignore[list-item]
            is_python_module=False,
            verbose=True,
        )
        _singular_triplet = LIB.singular_triplet  # pyright: ignore

        torch.utils.cpp_extension.load(
            name="singular_triplet",
            sources=[lib_base_path / "singular_triplet.cpp"],  # type: ignore[list-item]
            is_python_module=False,
            verbose=True,
        )
        _singular_triplet_debug = LIB.singular_triplet_debug  # pyright: ignore

    except Exception as exc:  # pylint: disable=broad-except
        warnings.warn(
            "Could not compile the custom binaries!"
            " Using torch.linalg.svd and torch.linalg.matrix_norm instead."
            f" Error: {exc}",
            UserWarning,
            stacklevel=2,
        )
        _singular_triplet = singular_triplet_native
        _singular_triplet_debug = singular_triplet_native
        _spectral_norm = spectral_norm_native
        _spectral_norm_debug = spectral_norm_native


def singular_triplet(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the singular triplet."""
    return _singular_triplet(A, u0, v0, maxiter, atol, rtol)


def singular_triplet_debug(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the singular triplet."""
    return _singular_triplet_debug(A, u0, v0, maxiter, atol, rtol)


def spectral_norm(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Tensor:
    """Computes the spectral norm."""
    return _spectral_norm(A, u0, v0, maxiter, atol, rtol)


def spectral_norm_debug(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Tensor:
    """Computes the spectral norm."""
    return _spectral_norm_debug(A, u0, v0, maxiter, atol, rtol)
