r"""Different Filter models to be used in conjunction with LinodeNet."""

__all__ = [
    # Types
    "Filter",
    # Constants
    "FILTERS",
    # Classes
    "KalmanCell",
    "KalmanFilter",
    "LinearKalmanCell",
    "NonLinearCell",
    "PseudoKalmanCell",
    "MissingValueFilter",
    "ResidualFilterBlock",
    "SequentialFilter",
    # imported
    "GRUFilter",
    "LSTMFilter",
    "RNNFilter",
]

from linodenet.models.filters._filters import (
    Filter,
    KalmanCell,
    KalmanFilter,
    LinearKalmanCell,
    NonLinearCell,
    PseudoKalmanCell,
)
from linodenet.models.filters._generic import (
    MissingValueFilter,
    ResidualFilterBlock,
    SequentialFilter,
)
from linodenet.models.filters._imported import GRUFilter, LSTMFilter, RNNFilter

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
