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
    # Constants
    "STATE_UPDATERS",
    # Types
    "StateUpdater",
    "StateUpdaterBase",
    # Classes
    "UpdateList",
    "UpdateSequence",
    "ResidualUpdateSequence",
    "MissingValueUpdate",
    "LinearUpdate",
    "LinearResidualUpdate",
    "NonLinearUpdate",
    "NonLinearKalmanUpdate",
    "KalmanUpdate",
    # Imported
    "RNN_Update",
    "GRU_Update",
    "LSTM_Update",
    "RNNCell",
    "GRUCell",
    "LSTMCell",
    # Functions
    "get_state_updater",
    "is_state_updater",
]

from torch.nn import GRUCell, LSTMCell, RNNCell

from . import probabilistic
from .base import (
    StateUpdater,
    StateUpdaterBase,
    get_state_updater,
    is_state_updater,
)
from .containers import (
    ResidualUpdateSequence,
    UpdateList,
    UpdateSequence,
)
from .imported import (
    GRU_Update,
    LSTM_Update,
    RNN_Update,
)
from .kalman_cell import (
    KalmanUpdate,
    NonLinearKalmanUpdate,
    NonLinearUpdate,
)
from .linear import (
    LinearResidualUpdate,
    LinearUpdate,
)
from .missing_value_filter import MissingValueUpdate

STATE_UPDATERS: dict[str, type[StateUpdater]] = {
    # PyTorch recurrent state updaters
    "GRU_Update": GRU_Update,
    "LSTM_Update": LSTM_Update,
    "RNN_Update": RNN_Update,
    # custom state updaters
    "KalmanUpdate": KalmanUpdate,
    "LinearUpdate": LinearUpdate,
    "LinearResidualUpdate": LinearResidualUpdate,
    "MissingValueUpdate": MissingValueUpdate,
    "NonLinearKalmanUpdate": NonLinearKalmanUpdate,
    "NonLinearUpdate": NonLinearUpdate,
    "UpdateSequence": UpdateSequence,
    "ResidualUpdateSequence": ResidualUpdateSequence,
}  # fmt: skip
r"""Dictionary of all available state updaters."""
