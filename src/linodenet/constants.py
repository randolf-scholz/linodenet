r"""Constants used throughout the library."""

__all__ = [
    "ATOL",
    "EMPTY_MAP",
    "EMPTY_SHAPE",
    "EPS",
    "FALSE",
    "NAN",
    "NEG_INF",
    "ONE",
    "POS_INF",
    "RTOL",
    "TRUE",
    "ZERO",
]


from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import BoolTensor, FloatTensor
from typing_extensions import Any, Final, Never

EMPTY_MAP: Final[Mapping[Any, Never]] = MappingProxyType({})
r"""CONSTANT: Immutable Empty dictionary."""
EMPTY_SHAPE: Final[torch.Size] = torch.Size([])
r"""CONSTANT: Empty shape."""
TRUE: Final[BoolTensor] = torch.tensor(True, dtype=torch.bool)  # type: ignore[assignment]
r"""A constant tensor representing the boolean value `True`."""
FALSE: Final[BoolTensor] = torch.tensor(False, dtype=torch.bool)  # type: ignore[assignment]
r"""A constant tensor representing the boolean value `False`."""
ZERO: Final[FloatTensor] = torch.tensor(0.0, dtype=torch.float32)  # type: ignore[assignment]
r"""A constant tensor representing the number `0`."""
ONE: Final[FloatTensor] = torch.tensor(1.0, dtype=torch.float32)  # type: ignore[assignment]
r"""A constant tensor representing the number `1`."""
NAN: Final[FloatTensor] = torch.tensor(float("nan"), dtype=torch.float32)  # type: ignore[assignment]
r"""A constant tensor representing the number `NaN`."""
POS_INF: Final[FloatTensor] = torch.tensor(float("inf"), dtype=torch.float32)  # type: ignore[assignment]
r"""A constant tensor representing the number `+∞`."""
NEG_INF: Final[FloatTensor] = torch.tensor(float("-inf"), dtype=torch.float32)  # type: ignore[assignment]
r"""A constant tensor representing the number `-∞`."""
ATOL: Final[float] = 1e-6
r"""CONST: Default absolute precision."""
RTOL: Final[float] = 1e-6
r"""CONST: Default relative precision."""
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
r"""CONST: Default epsilon for each dtype."""
