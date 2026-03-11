r"""Custom operators for the linodenet package."""

__all__ = [
    # CONSTANTS
    "ATOL",
    "RTOL",
    "BUILD_DIR",
    "LIB",
    "LIB_NAME",
    "SOURCE_DIR",
    # Protocols
    "KnownFunctions",
    # Implementations
    "singular_triplet",
    "singular_triplet_debug",
    "singular_triplet_riemann",
    "spectral_norm",
    "spectral_norm_debug",
    "spectral_norm_riemann",
    "ndtri_exp",
]

import os
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Optional, TypedDict, cast

import torch
from torch import Tensor

from linodenet_special.fallbacks.singular_triplet import SingularTriplet
from linodenet_special.fallbacks.spectral_norm import SpectralNorm

# constants
# we use FP32 machine epsilon as default tolerance
ATOL: Final[float] = 1e-6  # 2**-23  # ~1.19e-7
RTOL: Final[float] = 1e-6  # 2**-23  # ~1.19e-7
LIB_NAME: Final[str] = "liblinodenet_special"
r"""The name of the custom library."""
LIB: Final[ModuleType] = torch.ops.linodenet_special
r"""The custom library."""
BUILD_DIR: Final[Path] = Path(__file__).parent / "build"
r"""The build directory."""
SOURCE_DIR: Final[Path] = Path(__file__).parent / "src" / f"{LIB_NAME}"
r"""The source directory."""


class KnownFunctions(TypedDict):
    r"""The known functions in the custom library."""

    singular_triplet: SingularTriplet
    singular_triplet_debug: SingularTriplet
    singular_triplet_riemann: SingularTriplet
    spectral_norm: SpectralNorm
    spectral_norm_debug: SpectralNorm
    spectral_norm_riemann: SpectralNorm
    ndtri_exp: Callable[[Tensor], Tensor]


# region compile functions -------------------------------------------------------------
def _load_function(name: str, /) -> Any:
    r"""Load a function from the custom library."""
    from torch.utils import cpp_extension  # noqa: PLC0415

    try:  # compile the function
        print(f"Compiling {name}...", flush=True)
        cpp_extension.load(
            name=name,
            sources=[str(SOURCE_DIR / f"{name}.cpp")],
            extra_cflags=["-O3", "-DPy_LIMITED_API=0x030A0000"],
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
            verbose=True,
            with_cuda=torch.cuda.is_available(),
        )
    except Exception as exc:
        exc.add_note(f"Could not compile {name}!")
        raise

    try:  # load the function
        function = getattr(LIB, name)
    except AttributeError as exc:
        exc.add_note(f"Could not load {name}!")
        raise

    return function


def _compile_fns() -> KnownFunctions:
    r"""Fallback to compiling the functions."""
    from torch.utils import cpp_extension  # noqa: PLC0415

    cpp_extension.verify_ninja_availability()
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
    cache_dir = Path(
        os.environ.get("TORCH_EXTENSIONS_DIR", cpp_extension.get_default_build_root())
    )
    assert not cache_dir.exists() or cache_dir.is_dir()
    print("Compiling custom operators...")

    compiled_fns = {}
    exceptions = {}
    fn_names = KnownFunctions.__required_keys__

    for name in fn_names:
        try:
            compiled_fns[name] = _load_function(name)
        except Exception as _exc:
            exceptions[name] = _exc
    if exceptions:
        exc_group = ExceptionGroup("Failed to compile", list(exceptions.values()))
        error = RuntimeError(
            f"Failed to compile {len(exceptions)}/{len(fn_names)} custom operators!"
        )
        max_len = max(map(len, fn_names))
        FAILURE = "\033[91m❌️ FAILED\033[0m"
        SUCCESS = "\033[92m✅️ SUCCESS\033[0m"
        for name in fn_names:
            error.add_note(
                f"{name:<{max_len}}: {[SUCCESS, FAILURE][name in exceptions]}"
            )
        error.add_note(f"Consider clearing the torch_extension cache at {cache_dir!s}")
        raise error from exc_group

    return cast("KnownFunctions", compiled_fns)


def _load_linodenet() -> KnownFunctions:
    lib_file = BUILD_DIR / f"{LIB_NAME}.so"
    fn_names = KnownFunctions.__required_keys__

    if lib_file.exists():
        try:  # load pre-compiled binaries
            torch.ops.load_library(lib_file)
            # load the functions
            compiled_fns = {name: getattr(LIB, name) for name in fn_names}
            return cast("KnownFunctions", compiled_fns)
        except Exception as exc:
            warnings.warn(
                f"\n\t Custom binaries could not be loaded (raised {type(exc)!s})!"
                "\n\t Please ensure they are compiled for the correct platform."
                "\n\t Consider submitting a bug report.",
                UserWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            f"\n\t Custom binaries not found! ({lib_file!s})"
            "\n\t -> Consider compiling the linodenet_special extension.",
            UserWarning,
            stacklevel=2,
        )

    return _compile_fns()


_COMPILED_FNS: Final[KnownFunctions] = _load_linodenet()
r"""The compiled functions."""

_singular_triplet: SingularTriplet = _COMPILED_FNS["singular_triplet"]
_singular_triplet_debug: SingularTriplet = _COMPILED_FNS["singular_triplet_debug"]
_singular_triplet_riemann: SingularTriplet = _COMPILED_FNS["singular_triplet_riemann"]
_spectral_norm: SpectralNorm = _COMPILED_FNS["spectral_norm"]
_spectral_norm_debug: SpectralNorm = _COMPILED_FNS["spectral_norm_debug"]
_spectral_norm_riemann: SpectralNorm = _COMPILED_FNS["spectral_norm_riemann"]
ndtri_exp: Callable[[Tensor], Tensor] = _COMPILED_FNS["ndtri_exp"]
# endregion compile functions ----------------------------------------------------------


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
    return _spectral_norm(A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol)


def spectral_norm_debug(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    r"""Computes the spectral norm."""
    return _spectral_norm_debug(A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol)


def spectral_norm_riemann(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    r"""Computes the spectral norm."""
    return _spectral_norm_riemann(
        A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol
    )


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
    return _singular_triplet(A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol)


def singular_triplet_debug(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    return _singular_triplet_debug(
        A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol
    )


def singular_triplet_riemann(
    A: Tensor,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    return _singular_triplet_riemann(
        A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol
    )


# endregion singular triplet -----------------------------------------------------------
