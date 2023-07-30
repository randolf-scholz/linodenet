"""Constants used throughout the library."""

__all__ = ["TRUE", "FALSE", "ZERO", "ONE", "NAN", "POS_INF", "NEG_INF"]

from typing import Final

import torch
from torch import BoolTensor, FloatTensor

TRUE: Final[BoolTensor] = torch.tensor(True, dtype=torch.bool)  # type: ignore[assignment]
"""A constant tensor representing the boolean value ``True``."""
FALSE: Final[BoolTensor] = torch.tensor(False, dtype=torch.bool)  # type: ignore[assignment]
"""A constant tensor representing the boolean value ``False``."""
ZERO: Final[FloatTensor] = torch.tensor(0.0, dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number ``0``."""
ONE: Final[FloatTensor] = torch.tensor(1.0, dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number ``1``."""
NAN: Final[FloatTensor] = torch.tensor(float("nan"), dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number ``NaN``."""
POS_INF: Final[FloatTensor] = torch.tensor(float("inf"), dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number ``+∞``."""
NEG_INF: Final[FloatTensor] = torch.tensor(float("-inf"), dtype=torch.float32)  # type: ignore[assignment]
"""A constant tensor representing the number ``-∞``."""
