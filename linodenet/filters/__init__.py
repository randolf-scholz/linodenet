r"""Different Filter models to be used in conjunction with LinodeNet."""

__all__ = [
    # Types
    "Filter",
    # Constants
    "FILTERS",
    # Classes
    "FilterABC",
    "KalmanBlockCell",
    "KalmanCell",
    "KalmanFilter",
    "RecurrentCellFilter",
]


import logging
from typing import Final

from torch import nn

from linodenet.filters._filters import (
    FilterABC,
    KalmanBlockCell,
    KalmanCell,
    KalmanFilter,
    RecurrentCellFilter,
)

__logger__ = logging.getLogger(__name__)

Filter = nn.Module
"""Type hint for Filters"""

FILTERS: Final[dict[str, type[Filter]]] = {
    "FilterABc": FilterABC,
    "KalmanFilter": KalmanFilter,
    "KalmanCell": KalmanCell,
    "KalmanBlockCell": KalmanBlockCell,
    "RecurrentCellFilter": RecurrentCellFilter,
}
