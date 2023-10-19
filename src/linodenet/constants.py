"""Constants used throughout the library."""

__all__ = [
    "EMPTY_MAP",
    "TRUE",
    "FALSE",
    "ZERO",
    "ONE",
    "NAN",
    "POS_INF",
    "NEG_INF",
    "ATOL",
    "RTOL",
    "EPS",
]


from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import torch
from torch import BoolTensor, FloatTensor

EMPTY_MAP: Final[Mapping] = MappingProxyType({})
"""CONSTANT: Immutable Empty dictionary."""
TRUE: Final[BoolTensor] = torch.tensor(True, dtype=torch.bool)  # type: ignore[assignment]
"""A constant tensor representing the boolean value `True`."""
FALSE: Final[BoolTensor] = torch.tensor(False, dtype=torch.bool)  # type: ignore[assignment]
"""A constant tensor representing the boolean value `False`."""
ZERO: Final[FloatTensor] = torch.tensor(0.0, dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number `0`."""
ONE: Final[FloatTensor] = torch.tensor(1.0, dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number `1`."""
NAN: Final[FloatTensor] = torch.tensor(float("nan"), dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number `NaN`."""
POS_INF: Final[FloatTensor] = torch.tensor(float("inf"), dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number `+∞`."""
NEG_INF: Final[FloatTensor] = torch.tensor(float("-inf"), dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number `-∞`."""
ATOL: Final[float] = 1e-6
"""CONST: Default absolute precision."""
RTOL: Final[float] = 1e-6
"""CONST: Default relative precision."""
EPS: Final[dict[torch.dtype, float]] = {
    # other floats
    torch.bfloat16: 2**-7,  # ~7.81e-3
    # floats
    torch.float16: 2**-10,  # ~9.77e-4
    torch.float32: 2**-23,  # ~1.19e-7
    torch.float64: 2**-52,  # ~2.22e-16
    # complex floats
    torch.complex32: 2**-10,  # ~9.77e-4
    torch.complex64: 2**-23,  # ~1.19e-7
    torch.complex128: 2**-52,  # ~2.22e-16
}
"""CONST: Default epsilon for each dtype."""
