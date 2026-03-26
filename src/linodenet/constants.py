r"""Constants used throughout the library."""

__all__ = [
    # version
    "__version__",
    # Enums
    "FLOAT",
    # Constants
    "ATOL",
    "EMPTY_FN",
    "EMPTY_MAP",
    "EMPTY_SET",
    "EMPTY_SIZE",
    "EPS",
    "FALSE",
    "NAN",
    "NEG_INF",
    "ONE",
    "POS_INF",
    "RNG",
    "RTOL",
    "TRUE",
    "ZERO",
    "UNDEFINED",
]


import math
from collections.abc import Callable, Mapping
from enum import Enum
from importlib import metadata
from types import MappingProxyType
from typing import Any, Final, Never

import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor

try:  # single-source version
    __version__ = metadata.version(__package__ or __name__)
    r"""The version number of the `tsdm` package."""
except metadata.PackageNotFoundError:
    __version__ = "unknown"
    r"""The version number of the `tsdm` package."""


class FLOAT(float, Enum):
    r"""Enum: Common floating point values."""

    ZERO = 0.0
    ONE = 1.0
    INF = math.inf
    NAN = math.nan

    E = math.e
    PI = math.pi
    ROOT_2 = math.sqrt(2)
    ROOT_2PI = math.sqrt(2 * math.pi)
    ROOT_3 = math.sqrt(3)


# region collection constants ----------------------------------------------------------
EMPTY_MAP: Final[Mapping[Any, Never]] = MappingProxyType({})  # FIXME: PEP 603
r"""Constant: Immutable empty `Mapping`, use as default in function signatures."""
EMPTY_SET: Final[frozenset[Any]] = frozenset()
r"""Constant: Immutable empty `Set`, use as default in function signatures."""
EMPTY_SIZE: Final[torch.Size] = torch.Size([])
r"""Constant: Empty shape."""
EMPTY_FN: Final[Callable[..., None]] = lambda *_, **__: None  # noqa: E731
r"""Constant: Empty function, use as default in function signatures."""
UNDEFINED: Final[Any] = object()
r"""Constant: Sentinel value for unspecified arguments."""
# endregion collection constants -------------------------------------------------------


# region precision constants -----------------------------------------------------------
RNG: Final[Generator] = np.random.default_rng()
r"""Default random number generator."""
ATOL: Final[float] = 1e-6
r"""CONST: Default absolute precision."""
RTOL: Final[float] = 1e-6
r"""CONST: Default relative precision."""
EPS: Final[dict[torch.dtype, float]] = {
    torch.bfloat16   : 2**-7,   # ~7.81e-3
    torch.float16    : 2**-10,  # ~9.77e-4
    torch.float32    : 2**-23,  # ~1.19e-7
    torch.float64    : 2**-52,  # ~2.22e-16
    torch.complex32  : 2**-10,  # ~9.77e-4
    torch.complex64  : 2**-23,  # ~1.19e-7
    torch.complex128 : 2**-52,  # ~2.22e-16
}  # fmt: skip
r"""CONST: Default epsilon for each dtype."""
# endregion precision constants --------------------------------------------------------


# region tensor constants --------------------------------------------------------------
TRUE: Final[Tensor] = torch.tensor(True, dtype=torch.bool)
r"""A constant tensor representing the boolean value `True`."""
FALSE: Final[Tensor] = torch.tensor(False, dtype=torch.bool)
r"""A constant tensor representing the boolean value `False`."""
ZERO: Final[Tensor] = torch.tensor(0.0, dtype=torch.float32)
r"""A constant tensor representing the number `0`."""
ONE: Final[Tensor] = torch.tensor(1.0, dtype=torch.float32)
r"""A constant tensor representing the number `1`."""
NAN: Final[Tensor] = torch.tensor(float("nan"), dtype=torch.float32)
r"""A constant tensor representing the number `NaN`."""
POS_INF: Final[Tensor] = torch.tensor(float("inf"), dtype=torch.float32)
r"""A constant tensor representing the number `+∞`."""
NEG_INF: Final[Tensor] = torch.tensor(float("-inf"), dtype=torch.float32)
r"""A constant tensor representing the number `-∞`."""
# endregion tensor constants -----------------------------------------------------------
