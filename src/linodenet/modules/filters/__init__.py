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
    # ABCs & Protocols
    "Filter",
    "FilterBase",
    # Classes
    "MissingValueFilter",
    "ReZeroFilter",
    "ResNetFilter",
    "ResidualFilter",
    "SequentialFilter",
    # Functions
    "filter_from_config",
]

from torch.nn import GRUCell, LSTMCell, RNNCell

from linodenet.modules.filters import probabilistic
from linodenet.modules.filters.base import (
    FILTERS,
    Filter,
    FilterBase,
    MissingValueFilter,
    ResidualFilter,
    ResNetFilter,
    ReZeroFilter,
    SequentialFilter,
    filter_from_config,
)
from linodenet.modules.filters.filters import (
    KalmanFilter,
    LinearCell,
    LinearKalmanCell,
    LinearResidualCell,
    NonLinearCell,
    NonLinearKalmanCell,
    PseudoKalmanCell,
    ResidualCell,
    ResidualFilter,
    ResidualFilterBlock,
    SequentialFilter,
)

CELLS = {
    "GRUCell"            : GRUCell,
    "KalmanCell"         : NonLinearKalmanCell,
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
r"""Dictionary of all available cells (basic building blocks for filters)."""

FILTERS |= {
    "KalmanFilter"        : KalmanFilter,
    "MissingValueFilter"  : OldMissingValueFilter,
    "ProbabilisticFilter" : ProbabilisticFilter,
    "ResidualFilter"      : ResidualFilter,
    "ResidualFilterBlock" : ResidualFilterBlock,
    "SequentialFilter"    : SequentialFilter,
}  # fmt: skip
r"""Dictionary of all available filters."""
