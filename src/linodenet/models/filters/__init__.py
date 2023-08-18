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
    "LinearFilter",
    "NonLinearFilter",
    "PseudoKalmanFilter",
    "RecurrentCellFilter",
    "SequentialFilter",
    "SequentialFilterBlock",
]

from linodenet.models.filters._filters import (
    Filter,
    FilterABC,
    KalmanCell,
    KalmanFilter,
    LinearFilter,
    NonLinearFilter,
    PseudoKalmanFilter,
    RecurrentCellFilter,
    SequentialFilter,
    SequentialFilterBlock,
)

FILTERS: dict[str, type[Filter]] = {
    # "FilterABC": FilterABC,
    "KalmanCell": KalmanCell,
    "KalmanFilter": KalmanFilter,
    "LinearFilter": PseudoKalmanFilter,
    "NonLinearFilter": NonLinearFilter,
    "RecurrentCellFilter": RecurrentCellFilter,
    "SequentialFilter": SequentialFilter,
    "SequentialFilterBlock": SequentialFilterBlock,
}
r"""Dictionary of all available filters."""
