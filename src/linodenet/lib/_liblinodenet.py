"""Custom operators for the linodenet package."""

__all__ = [
    # CONSTANTS
    "BUILD_DIR",
    "CUSTOM_OPS",
    "LIB",
    "LIB_NAME",
    "SOURCE_DIR",
    # Protocols
    "SingularTriplet",
    "SpectralNorm",
    # Implementations
    "singular_triplet",
    "singular_triplet_debug",
    "singular_triplet_native",
    "singular_triplet_riemann",
    "spectral_norm",
    "spectral_norm_debug",
    "spectral_norm_native",
    "spectral_norm_riemann",
]

import warnings
from pathlib import Path

import torch
from torch import Tensor
from typing_extensions import Any, Optional, Protocol, runtime_checkable

# constants
# we use FP32 machine epsilon as default tolerance
ATOL = 1e-6  # 2**-23  # ~1.19e-7
RTOL = 1e-6  # 2**-23  # ~1.19e-7


# region Protocols ---------------------------------------------------------------------
@runtime_checkable
class SpectralNorm(Protocol):
    """Protocol for spectral norm implementations."""

    def __call__(
        self,
        A: Tensor,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
        maxiter: Optional[int] = None,
        atol: float = ATOL,
        rtol: float = RTOL,
    ) -> Tensor:
        """Computes the spectral norm.

        Args:
            A: The input matrix (shape: M×N).
            u0: The initial guess for the left singular vector (shape: M).
            v0: The initial guess for the right singular vector (shape: N).
            maxiter: The maximum number of iterations. (Default: O(M+N))
            atol: The absolute tolerance. (Default: 1e-6)
            rtol: The relative tolerance. (Default: 1e-6)

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
        atol: float = ATOL,
        rtol: float = RTOL,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Computes the singular triplet.

        Args:
            A: The input matrix (shape: M×N).
            u0: The initial guess for the left singular vector (shape: M).
            v0: The initial guess for the right singular vector (shape: N).
            maxiter: The maximum number of iterations. (Default: O(M+N))
            atol: The absolute tolerance. (Default: 1e-6)
            rtol: The relative tolerance. (Default: 1e-6)

        Returns:
            sigma: The singular value (scaler).
            u: The left singular vector (shape: M).
            v: The right singular vector (shape: N).
        """
        ...


# endregion ----------------------------------------------------------------------------


# region get compiled functions --------------------------------------------------------
LIB_NAME = "liblinodenet"
"""The name of the custom library."""
LIB = torch.ops.liblinodenet
"""The custom library."""
BUILD_DIR = Path(__file__).parent / "build"
"""The build directory."""
SOURCE_DIR = Path(__file__).parent / "src" / f"{LIB_NAME}"
"""The source directory."""
CUSTOM_OPS = [
    "singular_triplet",
    "singular_triplet_debug",
    "singular_triplet_riemann",
    "spectral_norm",
    "spectral_norm_debug",
    "spectral_norm_riemann",
]
"""List of custom operators."""


def load_function(name: str, /) -> Any:
    """Load a function from the custom library."""
    try:  # compile the function
        torch.utils.cpp_extension.load(  # pyright: ignore[reportAttributeAccessIssue]
            name=name,
            sources=[SOURCE_DIR / f"{name}.cpp"],  # type: ignore[list-item]
            is_python_module=False,
            verbose=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not compile {name}!") from exc

    try:  # load the function
        function = getattr(LIB, name)
    except AttributeError as exc:
        raise RuntimeError(f"Could not load {name}!") from exc

    return function


try:  # load pre-compiled binaries
    torch.ops.load_library(BUILD_DIR / f"{LIB_NAME}.so")
    # load the functions
    compiled_fns = {name: getattr(LIB, name) for name in CUSTOM_OPS}
except Exception as exc:
    warnings.warn(
        "Custom binaries not found! Trying to compile them on the fly!."
        " Please compile the extension in the linodenet/lib folder."
        f"\nFull error: {exc}\n{'-' * 80}",
        UserWarning,
        stacklevel=2,
    )
    compiled_fns = {name: load_function(name) for name in CUSTOM_OPS}

# fmt: off
_singular_triplet: SingularTriplet = compiled_fns["singular_triplet"]
_singular_triplet_debug: SingularTriplet = compiled_fns["singular_triplet_debug"]
_singular_triplet_riemann: SingularTriplet = compiled_fns["singular_triplet_riemann"]
_spectral_norm: SpectralNorm = compiled_fns["spectral_norm"]
_spectral_norm_debug: SpectralNorm = compiled_fns["spectral_norm_debug"]
_spectral_norm_riemann: SpectralNorm = compiled_fns["spectral_norm_riemann"]
# fmt: on

assert isinstance(_singular_triplet, SingularTriplet)
assert isinstance(_singular_triplet_debug, SingularTriplet)
assert isinstance(_singular_triplet_riemann, SingularTriplet)
assert isinstance(_spectral_norm, SpectralNorm)
assert isinstance(_spectral_norm_debug, SpectralNorm)
assert isinstance(_spectral_norm_riemann, SpectralNorm)
# endregion ----------------------------------------------------------------------------


# region spectral norm -----------------------------------------------------------------
def spectral_norm(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    """Computes the spectral norm."""
    return _spectral_norm(A, u0, v0, maxiter, atol, rtol)


def spectral_norm_debug(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    """Computes the spectral norm."""
    return _spectral_norm_debug(A, u0, v0, maxiter, atol, rtol)


def spectral_norm_riemann(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    """Computes the spectral norm."""
    return _spectral_norm_riemann(A, u0, v0, maxiter, atol, rtol)


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


# endregion ----------------------------------------------------------------------------


# region singular triplet --------------------------------------------------------------
def singular_triplet(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the singular triplet."""
    return _singular_triplet(A, u0, v0, maxiter, atol, rtol)


def singular_triplet_debug(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the singular triplet."""
    return _singular_triplet_debug(A, u0, v0, maxiter, atol, rtol)


def singular_triplet_riemann(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the singular triplet."""
    return _singular_triplet_riemann(A, u0, v0, maxiter, atol, rtol)


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


# endregion ----------------------------------------------------------------------------
