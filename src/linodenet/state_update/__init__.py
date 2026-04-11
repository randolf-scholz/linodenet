r"""State-update models to be used in conjunction with LinodeNet.

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
    "linear",
    "kalman",
    # Constants
    "STATE_UPDATERS",
    # Types
    "StateUpdater",
    "StateUpdaterBase",
    # Classes
    "StateUpdaterList",
    "CellSequence",
    "ResidualCellSequence",
    "ResidualCell",
    "MissingValueCell",
    "LinearRNNCell",
    "AttentionGain",
    "AttentionCovarianceFactor",
    "LinearCell",
    "KalmanCell",
    "NonLinearUpdate",
    "NonLinearKalmanUpdate",
    # Imported
    "RNNCell",
    "GRUCell",
    "LSTMCell",
    # Functions
    "is_state_updater",
]

from torch.nn import GRUCell, LSTMCell, RNNCell

from . import kalman, linear, probabilistic
from .base import (
    CellSequence,
    MissingValueCell,
    ResidualCell,
    ResidualCellSequence,
    StateUpdater,
    StateUpdaterBase,
    StateUpdaterList,
    is_state_updater,
)
from .kalman import NonLinearKalmanUpdate, NonLinearUpdate
from .linear import (
    AttentionCovarianceFactor,
    AttentionGain,
    KalmanCell,
    LinearCell,
    LinearRNNCell,
)

STATE_UPDATERS: dict[str, type[StateUpdater]] = {
    # PyTorch recurrent state updaters
    "GRUCell": GRUCell,
    "LSTMCell": LSTMCell,
    "RNNCell": RNNCell,
    # custom state updaters
    "LinearRNNCell": LinearRNNCell,
    "KalmanCell": KalmanCell,
    "LinearCell": LinearCell,
    "MissingValueCell": MissingValueCell,
    "NonLinearKalmanUpdate": NonLinearKalmanUpdate,
    "NonLinearUpdate": NonLinearUpdate,
    "CellSequence": CellSequence,
    "ResidualCell": ResidualCell,
    "ResidualCellSequence": ResidualCellSequence,
}  # fmt: skip
r"""Dictionary of all available state updaters."""
