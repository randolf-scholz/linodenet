r"""Different Filter models to be used in conjunction with LinodeNet."""

__all__ = [
    # Types
    "Filter",
    # Constants
    "FILTERS",
    # Classes
    "FilterABC",
    "KalmanCell",
    "KalmanFilter",
    "RecurrentCellFilter",
    "SequentialFilterBlock",
    "SequentialFilter",
]

from typing import Final, TypeAlias

from torch import nn

from linodenet.models.filters._filters import (
    FilterABC,
    KalmanCell,
    KalmanFilter,
    RecurrentCellFilter,
    SequentialFilter,
    SequentialFilterBlock,
)

Filter: TypeAlias = nn.Module
r"""Type hint for Filters"""

FILTERS: Final[dict[str, type[nn.Module]]] = {
    "FilterABc": FilterABC,
    "KalmanFilter": KalmanFilter,
    "KalmanCell": KalmanCell,
    "RecurrentCellFilter": RecurrentCellFilter,
}
r"""Dictionary of all available filters."""
