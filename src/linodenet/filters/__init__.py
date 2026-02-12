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
    # submodules
    "probabilistic",
    # Constants
    "FILTERS",
    "CELLS",
    # Types
    "Cell",
    "CellBase",
    "Filter",
    "FilterBase",
    # Classes
    "ReZeroFilter",
    "MissingValueCell",
    "CellList",
    "CellSequence",
    "ResidualCellSequence",
    # Cells
    "LinearCell",
    "LinearResidualCell",
    "NonLinearCell",
    "NonLinearKalmanCell",
    "PseudoKalmanCell",
    # Imported
    "RNNCell",
    "GRUCell",
    "LSTMCell",
    # Functions
    "get_filter",
]

from torch.nn import GRUCell, LSTMCell, RNNCell

from linodenet.filters import probabilistic
from linodenet.filters.base import (
    Cell,
    CellBase,
    Filter,
    FilterBase,
    get_filter,
)
from linodenet.filters.containers import CellList, CellSequence, ResidualCellSequence
from linodenet.filters.deprecated import PseudoKalmanCell, ReZeroFilter
from linodenet.filters.kalman_cell import (
    NonLinearCell,
    NonLinearKalmanCell,
)
from linodenet.filters.linear import (
    LinearCell,
    LinearResidualCell,
)
from linodenet.filters.missing_value_filter import MissingValueCell

CELLS: dict[str, type[Cell]]  = {
    # torch cells
    "GRUCell"            : GRUCell,
    "LSTMCell"           : LSTMCell,
    "RNNCell"            : RNNCell,
    # custom cells
    "KalmanCell"         : NonLinearKalmanCell,
    "LinearCell"         : LinearCell,
    "LinearResidualCell" : LinearResidualCell,
    "MissingValueCell"   : MissingValueCell,
    "NonLinearCell"      : NonLinearCell,
    "PseudoKalmanCell"   : PseudoKalmanCell,
}  # fmt: skip
r"""Dictionary of all available cells (basic building blocks for filters)."""

FILTERS: dict[str, type[Filter]] = {
    # "ProbabilisticFilter" : ProbabilisticFilter,
    # "ResidualFilterBlock" : ResidualFilterBlock,
}  # fmt: skip
r"""Dictionary of all available filters."""
