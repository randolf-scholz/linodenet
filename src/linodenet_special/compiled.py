r"""Custom operators for the linodenet package."""

__all__ = [
    # CONSTANTS
    "RAW_KERNELS",
    "AVAILABLE_KERNELS",
    # Implementations
    "bimodal_to_gaussian",
    "bimodal_to_gaussian_value_and_grad",
    "gaussian_to_bimodal",
    "gaussian_to_bimodal_value_and_grad",
    "gaussian_to_mixture",
    "gaussian_to_mixture_value_and_grad",
    "hard_bend",
    "mixture_to_gaussian",
    "mixture_to_gaussian_value_and_grad",
    "ndtri_exp",
    "singular_triplet",
    "spectral_norm",
]

import logging
import math
import os
import traceback
from collections.abc import Callable as Fn
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Optional, cast

import torch
from torch import Tensor

from .interfaces import (
    DEFAULT_NEWTON_MAXITER,
    DEFAULT_SPECTRAL_NORM_MAXITER,
    KnownFunctions,
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
_OPERATOR_SOURCE_FILES: Final[dict[str, str]] = {
    "singular_triplet": "singular_triplet.cpp",
    "spectral_norm": "spectral_norm.cpp",
    "ndtri_exp": "ndtri_exp.cpp",
    "hard_bend": "hard_bend.cpp",
    "bimodal_to_gaussian": "gaussian_transport.cpp",
    "bimodal_to_gaussian_value_and_grad": "gaussian_transport.cpp",
    "gaussian_to_bimodal": "gaussian_transport.cpp",
    "gaussian_to_bimodal_value_and_grad": "gaussian_transport.cpp",
    "gaussian_to_mixture": "gaussian_transport.cpp",
    "gaussian_to_mixture_value_and_grad": "gaussian_transport.cpp",
    "mixture_to_gaussian": "gaussian_transport.cpp",
    "mixture_to_gaussian_value_and_grad": "gaussian_transport.cpp",
}
r"""Mapping from exported operator names to their translation units."""

assert _SOURCE_DIR.is_dir(), f"{_SOURCE_DIR} is not a directory!"

logging.basicConfig(level=logging.WARNING)
__logger__ = logging.getLogger(__package__)


# region compile functions -------------------------------------------------------------
def _compile_fns() -> KnownFunctions:
    r"""Compile the available custom operators."""
    from torch.utils import cpp_extension  # noqa: PLC0415

    cpp_extension.verify_ninja_availability()
    cache_dir = Path(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            cpp_extension.get_default_build_root(),
        )
    )
    assert not cache_dir.exists() or cache_dir.is_dir()
    print("\nCompiling custom operators!")

    compiled_fns: dict[str, Any] = {}
    exceptions: dict[str, Exception] = {}

    for name in (fn_names := sorted(KnownFunctions.__required_keys__)):
        try:
            print(f"Compiling {name}...", flush=True)
            cpp_extension.load(
                name=name,
                sources=[str(_SOURCE_DIR / _OPERATOR_SOURCE_FILES[name])],
                extra_cflags=["-O3"],  # "-DPy_LIMITED_API=0x030E0000"],
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
        exception_details = "\n".join(
            f"\n{name}:\n{''.join(traceback.format_exception(exc)).rstrip()}"
            for name, exc in exceptions.items()
        )
        message = (
            f"\n{'-' * 80}\n"
            f"Exceptions:\n{exception_details}"
            f"\n{'-' * 80}\n"
            + "\n".join(
                f"{name:<{max_len}}: {[SUCCESS, FAILURE][name in exceptions]}"
                for name in fn_names
            )
            + f"\nFailed to compile {len(exceptions)}/{len(fn_names)} custom operators!"
            + f"\nConsider clearing the torch_extension cache at {cache_dir!s}"
            + f"\n{'-' * 80}\n"
        )
        __logger__.warning("%s", message)

    return cast(
        "KnownFunctions",
        {key: compiled_fns.get(key) for key in KnownFunctions.__required_keys__},
    )


def _load_prebuilts() -> KnownFunctions:
    r"""Load prebuilt binaries and return registered operators."""
    try:  # load pre-compiled binaries
        torch.ops.load_library(_LIB_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load custom binaries from {_LIB_FILE!s}."
        ) from exc

    return cast(
        "KnownFunctions",
        {key: getattr(_LIB, key, None) for key in KnownFunctions.__required_keys__},
    )


def _compile_liblinodenet() -> KnownFunctions:
    if _LIB_FILE.exists():
        compiled_fns = _load_prebuilts()
    else:
        __logger__.warning(
            f"\n\tCustom binaries not found! ({_LIB_FILE!s})"
            "\n\tConsider compiling the linodenet_special extension."
        )
        compiled_fns = _compile_fns()

    return compiled_fns


# region wrappers ----------------------------------------------------------------------


def gaussian_to_mixture(
    y: Tensor,
    /,
    weights: Tensor,
    mus: Tensor,
    sigmas: Tensor,
    *,
    maxiter: int | None = None,
) -> Tensor:
    r"""Optimal Transport from $N(0,1)$ to mixture $∑ₖωₖN(μₖ, σₖ²)$."""
    assert _gaussian_to_mixture is not None, "missing kernel"
    maxiter = DEFAULT_NEWTON_MAXITER[y.dtype] if maxiter is None else maxiter
    return _gaussian_to_mixture(y, weights, mus, sigmas, maxiter=maxiter)


def gaussian_to_mixture_value_and_grad(
    y: Tensor,
    /,
    weights: Tensor,
    mus: Tensor,
    sigmas: Tensor,
    *,
    maxiter: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Optimal transport from $N(0,1)$ to mixture $∑ₖωₖN(μₖ, σₖ²)$ and its derivative."""
    assert _gaussian_to_mixture_value_and_grad is not None, "missing kernel"
    maxiter = DEFAULT_NEWTON_MAXITER[y.dtype] if maxiter is None else maxiter
    return _gaussian_to_mixture_value_and_grad(y, weights, mus, sigmas, maxiter=maxiter)


def mixture_to_gaussian(
    x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> Tensor:
    r"""Optimal Transport from mixture $∑ₖωₖN(μₖ,σₖ²)$ to $N(0,1)$."""
    assert _mixture_to_gaussian is not None, "missing kernel"
    return _mixture_to_gaussian(x, weights, mus, sigmas)


def mixture_to_gaussian_value_and_grad(
    x: Tensor, /, weights: Tensor, mus: Tensor, sigmas: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Optimal transport from mixture $∑ₖωₖN(μₖ,σₖ²)$ to $N(0,1)$ and its derivative."""
    assert _mixture_to_gaussian_value_and_grad is not None, "missing kernel"
    return _mixture_to_gaussian_value_and_grad(x, weights, mus, sigmas)


def gaussian_to_bimodal(
    y: Tensor,
    /,
    mu: Tensor | float = 2.0,
    sigma: Tensor | float = 1.0,
    *,
    maxiter: int | None = None,
) -> Tensor:
    r"""Optimal Transport from $N(0, 1)$ to symmetric mixture $½N(-μ, σ²) + ½N(μ, σ²)$."""
    assert _gaussian_to_bimodal is not None, "missing kernel"
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    maxiter = DEFAULT_NEWTON_MAXITER[y.dtype] if maxiter is None else maxiter
    return _gaussian_to_bimodal(y, mu, sigma, maxiter=maxiter)


def gaussian_to_bimodal_value_and_grad(
    y: Tensor,
    /,
    mu: Tensor | float = 2.0,
    sigma: Tensor | float = 1.0,
    *,
    maxiter: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Optimal transport from $N(0,1)$ to symmetric mixture $½N(-μ, σ²) + ½N(μ, σ²)$ and its derivative."""
    assert _gaussian_to_bimodal_value_and_grad is not None, "missing kernel"
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    maxiter = DEFAULT_NEWTON_MAXITER[y.dtype] if maxiter is None else maxiter
    return _gaussian_to_bimodal_value_and_grad(y, mu, sigma, maxiter=maxiter)


def bimodal_to_gaussian(
    y: Tensor, /, mu: Tensor | float = 2.0, sigma: Tensor | float = 1.0
) -> Tensor:
    r"""Optimal Transport from mixture ½N(-μ, σ²) + ½N(μ, σ²) to N(0, 1)."""
    assert _bimodal_to_gaussian is not None, "missing kernel"
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    return _bimodal_to_gaussian(y, mu, sigma)


def bimodal_to_gaussian_value_and_grad(
    y: Tensor, /, mu: Tensor | float = 2.0, sigma: Tensor | float = 1.0
) -> tuple[Tensor, Tensor]:
    r"""Optimal transport from mixture $½N(-μ, σ²) + ½N(μ, σ²)$ to $N(0,1)$ and its derivative."""
    assert _bimodal_to_gaussian_value_and_grad is not None, "missing kernel"
    mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    return _bimodal_to_gaussian_value_and_grad(y, mu, sigma)


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
    maxiter: int | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tensor:
    r"""Computes the spectral norm."""
    assert _spectral_norm is not None, "missing kernel"
    maxiter = DEFAULT_SPECTRAL_NORM_MAXITER[A.dtype] if maxiter is None else maxiter
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
    maxiter: int | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    assert _singular_triplet is not None, "missing kernel"
    maxiter = DEFAULT_SPECTRAL_NORM_MAXITER[A.dtype] if maxiter is None else maxiter
    return _singular_triplet(A, u0=u0, v0=v0, maxiter=maxiter, atol=atol, rtol=rtol)


# endregion wrappers -------------------------------------------------------------------


RAW_KERNELS: Final[KnownFunctions] = _compile_liblinodenet()
# fmt: off
_bimodal_to_gaussian                : Fn | None  = RAW_KERNELS.get("bimodal_to_gaussian")
_bimodal_to_gaussian_value_and_grad : Fn | None  = RAW_KERNELS.get("bimodal_to_gaussian_value_and_grad")
_gaussian_to_bimodal                : Fn | None  = RAW_KERNELS.get("gaussian_to_bimodal")
_gaussian_to_bimodal_value_and_grad : Fn | None  = RAW_KERNELS.get("gaussian_to_bimodal_value_and_grad")
_gaussian_to_mixture                : Fn | None  = RAW_KERNELS.get("gaussian_to_mixture")
_gaussian_to_mixture_value_and_grad : Fn | None  = RAW_KERNELS.get("gaussian_to_mixture_value_and_grad")
_mixture_to_gaussian                : Fn | None  = RAW_KERNELS.get("mixture_to_gaussian")
_mixture_to_gaussian_value_and_grad : Fn | None  = RAW_KERNELS.get("mixture_to_gaussian_value_and_grad")
_hard_bend                          : Fn | None  = RAW_KERNELS.get("hard_bend")
_ndtri_exp                          : Fn | None  = RAW_KERNELS.get("ndtri_exp")
_singular_triplet                   : Fn | None  = RAW_KERNELS.get("singular_triplet")
_spectral_norm                      : Fn | None  = RAW_KERNELS.get("spectral_norm")
# fmt: on

AVAILABLE_KERNELS: Final[frozenset[str]] = frozenset(
    {
        name
        for name in KnownFunctions.__required_keys__
        if RAW_KERNELS.get(name) is not None
    }
)
