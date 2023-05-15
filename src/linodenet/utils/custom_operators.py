"""Custom operators for the linodenet package."""

import warnings
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

import linodenet.lib as lib

path = Path(lib.__path__[0]) / "build" / "liblinodenet.so"


if not path.exists():
    warnings.warn(
        "Custom binaries not found! Falling back to PyTorch implementation."
        " Please compile the extension in the linodenet/lib folder via poetry build.",
        UserWarning,
        stacklevel=2,
    )


def spectral_norm(
    A: torch.Tensor,
    u0: Optional[torch.Tensor] = None,
    v0: Optional[torch.Tensor] = None,
    max_iter: Optional[int] = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Tensor:
    """Computes the spectral norm."""
