r"""Custom operators for the linodenet package."""

__all__ = [
    # CONSTANTS
    "RAW_KERNELS",
    "WRAPPED_KERNELS",
    # Implementations
    "bimodal_to_gaussian",
    "gaussian_to_bimodal",
    "gaussian_to_mixture",
    "hard_bend",
    "mixture_to_gaussian",
    "ndtri_exp",
    "singular_triplet",
    "spectral_norm",
]

import logging
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Optional

import torch
from torch import Tensor

from .interfaces import (
    BimodalToGaussian,
    GaussianToBimodal,
    GaussianToMixture,
    HardBend,
    KnownFunctions,
    MixtureToGaussian,
    NdtriExp,
    SingularTriplet,
    SpectralNorm,
)

# constants
_LIB_NAME: Final[str] = "liblinodenet_special"
r"""The name of the custom library."""
_LIB: Final[ModuleType] = torch.ops.linodenet_special
r"""The custom library."""
_BUILD_DIR: Final[Path] = Path(__file__).parent / "build"
r"""The build directory."""
_SOURCE_DIR: Final[Path] = Path(__file__).parent / "csrc" / f"{_LIB_NAME}"
r"""The source directory."""
_LIB_FILE: Final[Path] = _BUILD_DIR / f"{_LIB_NAME}.so"
r"""The name of the custom library."""

assert _SOURCE_DIR.is_dir(), f"{_SOURCE_DIR} is not a directory!"

logging.basicConfig(level=logging.WARNING)
__logger__ = logging.getLogger(__package__)


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
                sources=[str(_SOURCE_DIR / f"{name}.cpp")],
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
            compiled_fns[name] = getattr(_LIB, name)

    if exceptions:
        max_len = max(map(len, fn_names))
        FAILURE = "\033[91m❌️ FAILED\033[0m"
        SUCCESS = "\033[92m✅️ SUCCESS\033[0m"

        message = (
            f"Failed to compile {len(exceptions)}/{len(fn_names)} custom operators!\n"
            + "\n".join(
                f"{name:<{max_len}}: {[SUCCESS, FAILURE][name in exceptions]}"
                for name in fn_names
            )
            + f"Consider clearing the torch_extension cache at {cache_dir!s}"
        )
        __logger__.warning("%s", message)

    return {
        "singular_triplet": compiled_fns.get("singular_triplet"),
        "spectral_norm": compiled_fns.get("spectral_norm"),
        "ndtri_exp": compiled_fns.get("ndtri_exp"),
        "hard_bend": compiled_fns.get("hard_bend"),
        "bimodal_to_gaussian": compiled_fns.get("bimodal_to_gaussian"),
        "gaussian_to_bimodal": compiled_fns.get("gaussian_to_bimodal"),
        "gaussian_to_mixture": compiled_fns.get("gaussian_to_mixture"),
        "mixture_to_gaussian": compiled_fns.get("mixture_to_gaussian"),
    }


def _load_prebuilts() -> KnownFunctions:
    r"""Load prebuilt binaries and return registered operators."""
    try:  # load pre-compiled binaries
        torch.ops.load_library(_LIB_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load custom binaries from {_LIB_FILE!s}."
        ) from exc

    return {
        "singular_triplet": getattr(_LIB, "singular_triplet", None),
        "spectral_norm": getattr(_LIB, "spectral_norm", None),
        "ndtri_exp": getattr(_LIB, "ndtri_exp", None),
        "hard_bend": getattr(_LIB, "hard_bend", None),
        "bimodal_to_gaussian": getattr(_LIB, "bimodal_to_gaussian", None),
        "gaussian_to_bimodal": getattr(_LIB, "gaussian_to_bimodal", None),
        "gaussian_to_mixture": getattr(_LIB, "gaussian_to_mixture", None),
        "mixture_to_gaussian": getattr(_LIB, "mixture_to_gaussian", None),
    }


def _compile_liblinodenet() -> KnownFunctions:
    if _LIB_FILE.exists():
        compiled_fns = _load_prebuilts()
    else:
        __logger__.warning(
            f"\n\t Custom binaries not found! ({_LIB_FILE!s})"
            "\n\t -> Consider compiling the linodenet_special extension."
        )
        compiled_fns = _compile_fns()

    return compiled_fns


# region wrappers ----------------------------------------------------------------------


def gaussian_to_mixture(
    y: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor, *, maxiter: int = 10
) -> Tensor:
    r"""Optimal Transport from $N(0,1)$ to mixture $∑ₖωₖN(μₖ, σₖ²)$."""
    assert _gaussian_to_mixture is not None, "missing kernel"
    return _gaussian_to_mixture(y, weights, mus, sigmas, maxiter=maxiter)


def mixture_to_gaussian(
    x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Optimal Transport from mixture $∑ₖωₖN(μₖ,σₖ²)$ to $N(0,1)$."""
    assert _mixture_to_gaussian is not None, "missing kernel"
    return _mixture_to_gaussian(x, weights, mus, sigmas)


def gaussian_to_bimodal(
    y: Tensor,
    /,
    mu: Tensor | float = 2.0,
    sigma: Tensor | float = 1.0,
    *,
    maxiter: int = 10,
) -> Tensor:
    r"""Optimal Transport from $N(0, 1)$ to symmetric mixture $½N(-μ, σ²) + ½N(μ, σ²)$."""
    assert _gaussian_to_bimodal is not None, "missing kernel"
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    return _gaussian_to_bimodal(y, mu, sigma, maxiter=maxiter)


def bimodal_to_gaussian(
    y: Tensor, /, mu: Tensor | float = 2.0, sigma: Tensor | float = 1.0
) -> Tensor:
    r"""Optimal Transport from mixture ½N(-μ, σ²) + ½N(μ, σ²) to N(0, 1)."""
    assert _bimodal_to_gaussian is not None, "missing kernel"
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    return _bimodal_to_gaussian(y, mu, sigma)


def ndtri_exp(log_p: Tensor, /) -> Tensor:
    r"""Inverse of `log_ndtr`, i.e. the log-quantile function of the standard normal distribution.

    torch currently does not implement the inverse of `log_ndtr`,
    this is simply a placeholder using the naive implementation.

    References:
        - scipy.special.ndtri_exp
    """
    assert _ndtri_exp is not None, "missing kernel"
    return _ndtri_exp(log_p)


def spectral_norm(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tensor:
    r"""Computes the spectral norm."""
    assert _spectral_norm is not None, "missing kernel"
    return _spectral_norm(A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol)


def hard_bend(
    x: Tensor,
    /,
    a: Tensor | float = math.e**2,
    c: Tensor | float = 2.0,
    m: Tensor | float = 1.0,
) -> Tensor:
    r"""Apply the hard bend activation function."""
    assert _hard_bend is not None, "missing kernel"
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
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    assert _singular_triplet is not None, "missing kernel"
    return _singular_triplet(A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol)


# endregion wrappers -------------------------------------------------------------------


RAW_KERNELS: Final[KnownFunctions] = _compile_liblinodenet()

# fmt: off
_bimodal_to_gaussian: BimodalToGaussian | None = RAW_KERNELS.get("bimodal_to_gaussian")
_gaussian_to_bimodal: GaussianToBimodal | None = RAW_KERNELS.get("gaussian_to_bimodal")
_gaussian_to_mixture: GaussianToMixture | None = RAW_KERNELS.get("gaussian_to_mixture")
_mixture_to_gaussian: MixtureToGaussian | None = RAW_KERNELS.get("mixture_to_gaussian")
_hard_bend:           HardBend          | None = RAW_KERNELS.get("hard_bend")
_ndtri_exp:           NdtriExp          | None = RAW_KERNELS.get("ndtri_exp")
_singular_triplet:    SingularTriplet   | None = RAW_KERNELS.get("singular_triplet")
_spectral_norm:       SpectralNorm      | None = RAW_KERNELS.get("spectral_norm")
# fmt: on


WRAPPED_KERNELS: Final[KnownFunctions] = {
    "bimodal_to_gaussian" : None if RAW_KERNELS.get("bimodal_to_gaussian") is None else bimodal_to_gaussian,
    "gaussian_to_bimodal" : None if RAW_KERNELS.get("gaussian_to_bimodal") is None else gaussian_to_bimodal,
    "gaussian_to_mixture" : None if RAW_KERNELS.get("gaussian_to_mixture") is None else gaussian_to_mixture,
    "mixture_to_gaussian" : None if RAW_KERNELS.get("mixture_to_gaussian") is None else mixture_to_gaussian,
    "hard_bend"           : None if RAW_KERNELS.get("hard_bend")           is None else hard_bend,
    "ndtri_exp"           : None if RAW_KERNELS.get("ndtri_exp")           is None else ndtri_exp,
    "singular_triplet"    : None if RAW_KERNELS.get("singular_triplet")    is None else singular_triplet,
    "spectral_norm"       : None if RAW_KERNELS.get("spectral_norm")       is None else spectral_norm,
}  # fmt: skip
r"""Wrapped C++ kernels."""
