r"""Filter models to be used in conjunction with LinodeNet.

Background
----------

The filtering problem is a fundamental problem in signal processing and control theory.
Consider the following setup:

1. (stochastic) Dynamical System: $x(t+∆t) = Φ(xₜ, ∆t)$
2. (stochastic) Measurement System: $y(t) = H(x(t)) + ε(t)$

Then, given observations $\{y_{t₁}, y_{t₂}, …, y_{tₙ}\}$, we want to estimate the
state $\{x_{q₁}, x_{q₂}, …, x_{qₙ}\}$ at the query times $\{q₁, q₂, …, qₙ\}$.
"""

__all__ = [
    # Constants
    "CELLS",
    "FILTERS",
    # ABCs & Protocols
    "Cell",
    "Filter",
    "ProbabilisticFilter",
    "FilterABC",
    # Classes
    "MissingValueFilter",
    "GRUCell",
    "KalmanCell",
    "LSTMCell",
    "LinearCell",
    "LinearKalmanCell",
    "LinearResidualCell",
    "MissingValueCell",
    "NonLinearCell",
    "PseudoKalmanCell",
    "RNNCell",
    "ResidualCell",
    "KalmanFilter",
    "ResidualFilter",
    "ResidualFilterBlock",
    "SequentialFilter",
]

from linodenet.modules.filters.base import (
    Cell,
    Filter,
    FilterABC,
    MissingValueFilter,
    ProbabilisticFilter,
)
from linodenet.modules.filters.cells import (
    GRUCell,
    KalmanCell,
    LinearCell,
    LinearKalmanCell,
    LinearResidualCell,
    LSTMCell,
    MissingValueCell,
    NonLinearCell,
    PseudoKalmanCell,
    ResidualCell,
    RNNCell,
)
from linodenet.modules.filters.filters import (
    KalmanFilter,
    ResidualFilter,
    ResidualFilterBlock,
    SequentialFilter,
)

CELLS: dict[str, type[Cell]] = {
    "GRUCell"            : GRUCell,
    "KalmanCell"         : KalmanCell,
    "LinearCell"         : LinearCell,
    "LinearKalmanCell"   : LinearKalmanCell,
    "LinearResidualCell" : LinearResidualCell,
    "LSTMCell"           : LSTMCell,
    "MissingValueCell"   : MissingValueCell,
    "NonLinearCell"      : NonLinearCell,
    "PseudoKalmanCell"   : PseudoKalmanCell,
    "ResidualCell"       : ResidualCell,
    "RNNCell"            : RNNCell,
}  # fmt: skip
"""Dictionary of all available cells (basic building blocks for filters)."""

FILTERS: dict[str, type[Filter]] = {
    "KalmanFilter"        : KalmanFilter,
    "MissingValueFilter"  : MissingValueFilter,
    "ProbabilisticFilter" : ProbabilisticFilter,
    "ResidualFilter"      : ResidualFilter,
    "ResidualFilterBlock" : ResidualFilterBlock,
    "SequentialFilter"    : SequentialFilter,
}  # fmt: skip
r"""Dictionary of all available filters."""
