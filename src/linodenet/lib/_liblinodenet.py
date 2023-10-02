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
from typing import Any, Optional, Protocol, cast, runtime_checkable

import torch
import torch.utils.cpp_extension
from torch import Tensor


@runtime_checkable
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
        """Computes the spectral norm.

        Args:
            A: The input matrix (shape: M×N).
            u0: The initial guess for the left singular vector (shape: M).
            v0: The initial guess for the right singular vector (shape: N).
            maxiter: The maximum number of iterations. (Default: 4⋅(M+N) + 100)
            atol: The absolute tolerance. (Default: 1e-8)
            rtol: The relative tolerance. (Default: 1e-5)

        Returns:
            sigma: The singular value (scaler).
        """
        ...


@runtime_checkable
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
        """Computes the singular triplet.

        Args:
            A: The input matrix (shape: M×N).
            u0: The initial guess for the left singular vector (shape: M).
            v0: The initial guess for the right singular vector (shape: N).
            maxiter: The maximum number of iterations. (Default: 4⋅(M+N) + 100)
            atol: The absolute tolerance. (Default: 1e-8)
            rtol: The relative tolerance. (Default: 1e-5)

        Returns:
            sigma: The singular value (scaler).
            u: The left singular vector (shape: M).
            v: The right singular vector (shape: N).
        """
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
    U, S, Vh = torch.linalg.svd(A)
    # cols of U = LSV, rows of Vh: RSV
    return S[0], U[:, 0], Vh[0, :]


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


def load_function(name: str, /) -> Any:
    """Load a function from the custom library."""
    # compile the function
    try:
        torch.utils.cpp_extension.load(
            name=name,
            sources=[source_dir / f"{name}.cpp"],  # type: ignore[list-item]
            is_python_module=False,
            verbose=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Could not compile {name}! Error: {exc}")

    # load the function
    try:
        function = getattr(LIB, name)
    except AttributeError as exc:
        raise RuntimeError(f"Could not load {name}! Error: {exc}")

    return function


lib_base_path = Path(__file__).parent
build_dir = lib_base_path / "build" / "liblinodenet.so"
source_dir = lib_base_path / "src" / "liblinodenet"

_singular_triplet: SingularTriplet
_singular_triplet_debug: SingularTriplet
_spectral_norm: SpectralNorm
_spectral_norm_debug: SpectralNorm

LIB = torch.ops.liblinodenet
"""The custom library."""

if build_dir.exists():
    torch.ops.load_library(build_dir)
    _singular_triplet = cast(SingularTriplet, LIB.singular_triplet)
    _singular_triplet_debug = cast(SingularTriplet, LIB.singular_triplet_debug)
    _spectral_norm = cast(SpectralNorm, LIB.spectral_norm)
    _spectral_norm_debug = cast(SpectralNorm, LIB.spectral_norm_debug)
else:
    warnings.warn(
        "Custom binaries not found! Trying to compile them on the fly!."
        " Please compile the extension in the linodenet/lib folder via poetry build.",
        UserWarning,
        stacklevel=2,
    )
    compiled_fns = {
        name: load_function(name)
        for name in (
            "spectral_norm",
            "singular_triplet",
            "singular_triplet_debug",
            "spectral_norm_debug",
        )
    }
    _singular_triplet_debug = compiled_fns["singular_triplet_debug"]
    _singular_triplet = compiled_fns["singular_triplet"]
    _spectral_norm = compiled_fns["spectral_norm"]
    _spectral_norm_debug = compiled_fns["spectral_norm_debug"]

    # _singular_triplet = singular_triplet_native
    # _singular_triplet_debug = singular_triplet_native
    # _spectral_norm = spectral_norm_native
    # _spectral_norm_debug = spectral_norm_native

assert isinstance(_singular_triplet, SingularTriplet)
assert isinstance(_singular_triplet_debug, SingularTriplet)
assert isinstance(_spectral_norm, SpectralNorm)
assert isinstance(_spectral_norm_debug, SpectralNorm)


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
