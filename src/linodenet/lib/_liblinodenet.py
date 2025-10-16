r"""Custom operators for the linodenet package."""
# ruff: noqa: ARG001

__all__ = [
    # CONSTANTS
    "ATOL",
    "RTOL",
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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.utils import cpp_extension

# constants
# we use FP32 machine epsilon as default tolerance
ATOL: Final[float] = 1e-6  # 2**-23  # ~1.19e-7
RTOL: Final[float] = 1e-6  # 2**-23  # ~1.19e-7
LIB_NAME = "liblinodenet"
r"""The name of the custom library."""
LIB = torch.ops.liblinodenet
r"""The custom library."""
BUILD_DIR = Path(__file__).parent / "build"
r"""The build directory."""
SOURCE_DIR = Path(__file__).parent / "src" / f"{LIB_NAME}"
r"""The source directory."""
CUSTOM_OPS = [
    "singular_triplet",
    "singular_triplet_debug",
    "singular_triplet_riemann",
    "spectral_norm",
    "spectral_norm_debug",
    "spectral_norm_riemann",
]
r"""List of custom operators."""


# region compile functions -------------------------------------------------------------
def load_function(name: str, /) -> Any:
    r"""Load a function from the custom library."""
    try:  # compile the function
        cpp_extension.load(
            name=name,
            sources=[SOURCE_DIR / f"{name}.cpp"],  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]
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


def _load_linodenet() -> dict[str, Callable]:
    def _compile_fns() -> dict[str, Callable]:
        r"""Fallback to compiling the functions."""
        compiled_fns = {}
        exceptions = {}
        for name in CUSTOM_OPS:
            try:
                compiled_fns[name] = load_function(name)
            except Exception as _exc:  # noqa: BLE001
                exceptions[name] = _exc
        if exceptions:
            exc_group = ExceptionGroup("Failed to compile", list(exceptions.values()))
            error = RuntimeError(
                f"Failed to compile {len(exceptions)}/{len(CUSTOM_OPS)} custom operators!"
            )
            max_len = max(map(len, CUSTOM_OPS))
            FAILURE = "\033[91m✘ FAILED\033[0m"
            SUCCESS = "\033[92m✔ SUCCESS\033[0m"
            for name in CUSTOM_OPS:
                error.add_note(
                    f"{name:<{max_len}}: {[SUCCESS, FAILURE][name in exceptions]}"
                )
            raise error from exc_group
        return compiled_fns

    try:  # load pre-compiled binaries
        torch.ops.load_library(BUILD_DIR / f"{LIB_NAME}.so")
        # load the functions
        return {name: getattr(LIB, name) for name in CUSTOM_OPS}
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            "Custom binaries not found!"
            "\n\t-> Trying to compile them on the fly!."
            "\n\t-> Consider compiling the extension in the linodenet/lib folder."
            f"\n\t-> Full error: {exc}"
            f"\n{'-' * 80}",
            UserWarning,
            stacklevel=2,
        )
        return _compile_fns()


COMPILED_FNS = _load_linodenet()
r"""The compiled functions."""
# endregion compile functions ----------------------------------------------------------


# region protocols ---------------------------------------------------------------------
@runtime_checkable
class SpectralNorm(Protocol):
    r"""Protocol for spectral norm implementations."""

    def __call__(
        self,
        A: Tensor,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
        maxiter: Optional[int] = None,
        atol: float = ATOL,
        rtol: float = RTOL,
    ) -> Tensor:
        r"""Computes the spectral norm.

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
    r"""Protocol for singular triplet implementations."""

    def __call__(
        self,
        A: Tensor,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
        maxiter: Optional[int] = None,
        atol: float = ATOL,
        rtol: float = RTOL,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Computes the singular triplet.

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


_singular_triplet: SingularTriplet = COMPILED_FNS["singular_triplet"]
_singular_triplet_debug: SingularTriplet = COMPILED_FNS["singular_triplet_debug"]
_singular_triplet_riemann: SingularTriplet = COMPILED_FNS["singular_triplet_riemann"]
_spectral_norm: SpectralNorm = COMPILED_FNS["spectral_norm"]
_spectral_norm_debug: SpectralNorm = COMPILED_FNS["spectral_norm_debug"]
_spectral_norm_riemann: SpectralNorm = COMPILED_FNS["spectral_norm_riemann"]
# endregion protocols ------------------------------------------------------------------


# region spectral norm -----------------------------------------------------------------
def spectral_norm(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    r"""Computes the spectral norm."""
    return _spectral_norm(A, u0, v0, maxiter, atol, rtol)


def spectral_norm_debug(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    r"""Computes the spectral norm."""
    return _spectral_norm_debug(A, u0, v0, maxiter, atol, rtol)


def spectral_norm_riemann(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    r"""Computes the spectral norm."""
    return _spectral_norm_riemann(A, u0, v0, maxiter, atol, rtol)


def spectral_norm_native(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Tensor:
    r"""Computes the spectral norm."""
    return torch.linalg.matrix_norm(A, ord=2)


# endregion spectral norm --------------------------------------------------------------


# region singular triplet --------------------------------------------------------------
def singular_triplet(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    return _singular_triplet(A, u0, v0, maxiter, atol, rtol)


def singular_triplet_debug(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    return _singular_triplet_debug(A, u0, v0, maxiter, atol, rtol)


def singular_triplet_riemann(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    return _singular_triplet_riemann(A, u0, v0, maxiter, atol, rtol)


def singular_triplet_native(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    U, S, Vh = torch.linalg.svd(A)
    # cols of U = LSV, rows of Vh: RSV
    return S[0], U[:, 0], Vh[0, :]


# endregion singular triplet -----------------------------------------------------------
