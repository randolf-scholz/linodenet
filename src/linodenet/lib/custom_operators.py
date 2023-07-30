"""Custom operators for the linodenet package."""

__all__ = [
    "singular_triplet",
    "spectral_norm",
    "singular_triplet_native",
    "spectral_norm_native",
]

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import torch.utils.cpp_extension
from torch import Tensor


def singular_triplet_native(
    A: torch.Tensor,
    u0: Optional[torch.Tensor] = None,
    v0: Optional[torch.Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the singular triplet."""
    U, S, V = torch.linalg.svd(A)
    u, s, v = U[:, 0], S[0], V[0, :]
    return s, u, v


def spectral_norm_native(
    A: torch.Tensor,
    u0: Optional[torch.Tensor] = None,
    v0: Optional[torch.Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Tensor:
    """Computes the spectral norm."""
    return torch.linalg.matrix_norm(A, ord=2)


lib_base_path = Path(__file__).parent
lib_path = lib_base_path / "build" / "liblinodenet.so"

_singular_triplet: Callable[
    [Tensor, Optional[Tensor], Optional[Tensor], Optional[int], float, float],
    tuple[Tensor, Tensor, Tensor],
]
_spectral_norm: Callable[
    [Tensor, Optional[Tensor], Optional[Tensor], Optional[int], float, float], Tensor
]

if lib_path.exists():
    torch.ops.load_library(lib_path)
    _spectral_norm = torch.ops.custom.spectral_norm  # pyright: ignore
    _singular_triplet = torch.ops.custom.singular_triplet  # pyright: ignore
else:
    warnings.warn(
        "Custom binaries not found! Trying to compile them on the fly!."
        " Please compile the extension in the linodenet/lib folder via poetry build.",
        UserWarning,
        stacklevel=2,
    )

    try:
        torch.utils.cpp_extension.load(
            name="singular_triplet",
            sources=[lib_base_path / "singular_triplet.cpp"],  # type: ignore[list-item]
            is_python_module=False,
            verbose=True,
        )
        _singular_triplet = torch.ops.custom.singular_triplet  # pyright: ignore

        torch.utils.cpp_extension.load(
            name="spectral_norm",
            sources=[lib_base_path / "spectral_norm.cpp"],  # type: ignore[list-item]
            is_python_module=False,
            verbose=True,
        )
        _spectral_norm = torch.ops.custom.spectral_norm  # pyright: ignore
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn(
            "Could not compile the custom binaries!"
            " Using torch.linalg.svd and torch.linalg.matrix_norm instead."
            f" Error: {e}",
            UserWarning,
            stacklevel=2,
        )
        _singular_triplet = singular_triplet_native
        _spectral_norm = spectral_norm_native


def singular_triplet(
    A: torch.Tensor,
    u0: Optional[torch.Tensor] = None,
    v0: Optional[torch.Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the singular triplet."""
    return _singular_triplet(A, u0, v0, maxiter, atol, rtol)


def spectral_norm(
    A: torch.Tensor,
    u0: Optional[torch.Tensor] = None,
    v0: Optional[torch.Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Tensor:
    """Computes the spectral norm."""
    return _spectral_norm(A, u0, v0, maxiter, atol, rtol)
