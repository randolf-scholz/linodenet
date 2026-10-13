r"""Constants used throughout the library."""

__all__ = [
    # Constants
    "__version__",
    "ATOL",
    "EPS",
    "RTOL",
]


from importlib import metadata
from typing import Final

import torch

__version__: Final[str] = "0.5.2"
r"""The version number of the package."""

if metadata.version(__package__ or __name__) != __version__:
    raise ValueError(f"pyproject.toml version does not match {__version__=!r}")


ATOL: Final[float] = 1e-6
r"""Default absolute tolerance for numerical comparisons."""
RTOL: Final[float] = 1e-6
r"""Default relative tolerance for numerical comparisons."""
EPS: Final[dict[torch.dtype, float]] = {
    torch.bfloat16   : 2**-7,   # ~7.81e-3
    torch.float16    : 2**-10,  # ~9.77e-4
    torch.float32    : 2**-23,  # ~1.19e-7
    torch.float64    : 2**-52,  # ~2.22e-16
    torch.complex32  : 2**-10,  # ~9.77e-4
    torch.complex64  : 2**-23,  # ~1.19e-7
    torch.complex128 : 2**-52,  # ~2.22e-16
}  # fmt: skip
r"""Machine epsilon for the supported floating-point and complex dtypes.

For complex dtypes, use the epsilon of the corresponding real component dtype.
"""
