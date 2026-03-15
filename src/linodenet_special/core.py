r"""Custom operators for the linodenet package."""

__all__ = [
    # CONSTANTS
    "ATOL",
    "RTOL",
    "BUILD_DIR",
    "COMPILED",
    "KERNELS",
    "LIB",
    "LIB_NAME",
    "LIB_FILE",
    "SOURCE_DIR",
    # Protocols
    "KnownFunctions",
    "Kernels",
    # Implementations
    "singular_triplet",
    "spectral_norm",
    "ndtri_exp",
    "hard_bend",
]

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Optional, TypedDict

import torch
from torch import Tensor

from linodenet_special.fallbacks import FALLBACKS
from linodenet_special.fallbacks.hard_bend import HardBend
from linodenet_special.fallbacks.ndtri_exp import NdtriExp
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
SOURCE_DIR: Final[Path] = Path(__file__).parent / "csrc" / f"{LIB_NAME}"
r"""The source directory."""
LIB_FILE: Final[Path] = BUILD_DIR / f"{LIB_NAME}.so"
r"""The name of the custom library."""

assert SOURCE_DIR.is_dir(), f"{SOURCE_DIR} is not a directory!"

logging.basicConfig(level=logging.WARNING)
__logger__ = logging.getLogger(__package__)


class KnownFunctions(TypedDict):
    r"""The known functions in the custom library."""

    singular_triplet: SingularTriplet | None
    spectral_norm: SpectralNorm | None
    ndtri_exp: NdtriExp | None
    hard_bend: HardBend | None


@dataclass(frozen=True)
class Kernels:
    r"""The selected kernels exposed as attributes."""

    singular_triplet: SingularTriplet
    spectral_norm: SpectralNorm
    ndtri_exp: NdtriExp
    hard_bend: HardBend


# region compile functions -------------------------------------------------------------
def _compile_fns() -> KnownFunctions:
    r"""Compile the available custom operators."""
    from torch.utils import cpp_extension  # noqa: PLC0415

    cpp_extension.verify_ninja_availability()
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
    cache_dir = Path(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            cpp_extension.get_default_build_root(),
        )
    )
    assert not cache_dir.exists() or cache_dir.is_dir()
    print("Compiling custom operators...")

    compiled_fns: dict[str, Any] = {}
    exceptions: dict[str, Exception] = {}

    for name in (fn_names := KnownFunctions.__required_keys__):
        try:
            print(f"Compiling {name}...", flush=True)
            cpp_extension.load(
                name=name,
                sources=[str(SOURCE_DIR / f"{name}.cpp")],
                extra_cflags=["-O3"],  # , "-DPy_LIMITED_API=0x030A0000"],
                extra_cuda_cflags=["-O3"],
                is_python_module=False,
                verbose=False,
                with_cuda=torch.cuda.is_available(),
            )
        except Exception as _exc:
            _exc.add_note(f"Could not compile {name}!")
            exceptions[name] = _exc
        else:
            compiled_fns[name] = getattr(LIB, name)

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

    return {
        "singular_triplet": compiled_fns.get("singular_triplet"),
        "spectral_norm": compiled_fns.get("spectral_norm"),
        "ndtri_exp": compiled_fns.get("ndtri_exp"),
        "hard_bend": compiled_fns.get("hard_bend"),
    }


def _load_prebuilts() -> KnownFunctions:
    r"""Load prebuilt binaries and return registered operators."""
    try:  # load pre-compiled binaries
        torch.ops.load_library(LIB_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load custom binaries from {LIB_FILE!s}."
        ) from exc

    return {
        "singular_triplet": getattr(LIB, "singular_triplet", None),
        "spectral_norm": getattr(LIB, "spectral_norm", None),
        "ndtri_exp": getattr(LIB, "ndtri_exp", None),
        "hard_bend": getattr(LIB, "hard_bend", None),
    }


def _load_linodenet() -> KnownFunctions:
    if LIB_FILE.exists():
        compiled_fns = _load_prebuilts()
    else:
        __logger__.warning(
            f"\n\t Custom binaries not found! ({LIB_FILE!s})"
            "\n\t -> Consider compiling the linodenet_special extension."
        )
        compiled_fns = _compile_fns()

    return compiled_fns


def _select_fns(compiled_fns: KnownFunctions, /) -> Kernels:
    r"""Select compiled kernels when available and fall back otherwise."""
    impls: dict[str, Any] = {}
    missing: set[str] = set()
    for name in KnownFunctions.__required_keys__:
        if (compiled_fn := compiled_fns.get(name)) is not None:
            impls[name] = compiled_fn
        else:
            impls[name] = FALLBACKS[name]
            missing.add(name)

    if missing:
        __logger__.warning(
            "Missing compiled versions of the following kernels:"
            f"\n\t- {'\n\t- '.join(missing)}"
            "\nUsing pure python fallbacks for these functions."
        )

    return Kernels(**impls)


COMPILED: Final[KnownFunctions] = _load_linodenet()
r"""The compiled functions that were successfully loaded."""

KERNELS: Final[Kernels] = _select_fns(COMPILED)
r"""The selected kernels, using compiled functions and fallback implementations."""

_singular_triplet = KERNELS.singular_triplet
_spectral_norm = KERNELS.spectral_norm
_ndtri_exp = KERNELS.ndtri_exp
_hard_bend = KERNELS.hard_bend


def ndtri_exp(log_p: Tensor, /) -> Tensor:
    r"""Inverse of `log_ndtr`, i.e. the log-quantile function of the standard normal distribution.

    torch currently does not implement the inverse of `log_ndtr`,
    this is simply a placeholder using the naive implementation.

    References:
        - scipy.special.ndtri_exp
    """
    return _ndtri_exp(log_p)


def spectral_norm(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    r"""Computes the spectral norm."""
    return _spectral_norm(A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol)


def hard_bend(
    x: Tensor,
    /,
    *,
    a: Tensor | float = math.e**2,
    c: Tensor | float = 2.0,
    m: Tensor | float = 1.0,
) -> Tensor:
    r"""Apply the hard bend activation function."""
    a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    m = torch.as_tensor(m, dtype=x.dtype, device=x.device)
    return _hard_bend(x, a, c, m)


def singular_triplet(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    return _singular_triplet(A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol)
