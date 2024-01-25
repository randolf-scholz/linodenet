r"""Different Filter models to be used in conjunction with LinodeNet."""

__all__ = [
    # Constants
    "FILTERS",
    # ABCs & Protocols
    "Filter",
    # Classes
    "GRUFilter",
    "KalmanCell",
    "KalmanFilter",
    "LSTMFilter",
    "LinearKalmanCell",
    "MissingValueFilter",
    "NonLinearCell",
    "PseudoKalmanCell",
    "RNNFilter",
    "ResidualFilterBlock",
    "SequentialFilter",
]

from linodenet.models.filters._generic import (
    MissingValueFilter,
    ResidualFilterBlock,
    SequentialFilter,
)
from linodenet.models.filters._imported import GRUFilter, LSTMFilter, RNNFilter
from linodenet.models.filters.base import (
    Filter,
    KalmanCell,
    KalmanFilter,
    LinearKalmanCell,
    NonLinearCell,
    PseudoKalmanCell,
)

FILTERS: dict[str, type[Filter]] = {
    "KalmanCell": KalmanCell,
    "KalmanFilter": KalmanFilter,
    "LinearFilter": PseudoKalmanCell,
    "NonLinearFilter": NonLinearCell,
    "RecurrentCellFilter": MissingValueFilter,
    "SequentialFilter": SequentialFilter,
    "SequentialFilterBlock": ResidualFilterBlock,
    # imported
    "GRUFilter": GRUFilter,
    "LSTMFilter": LSTMFilter,
    "RNNFilter": RNNFilter,
}
r"""Dictionary of all available filters."""
