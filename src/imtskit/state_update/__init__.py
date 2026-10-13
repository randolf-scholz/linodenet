r"""State-update models for assimilating observations into state estimates.

## Background

Given a dynamical system and an observation model,

$$\begin{aligned}
    x(t+∆t) &= Φ(x(t), ∆t), \\
    y(t) &= H(x(t)) + ε(t),
\end{aligned}$$

state estimation uses observations $y(t₁), …, y(tₙ)$ to estimate states
$x(q₁), …, x(qₘ)$ at requested query times $q₁, …, qₘ$. The state-update
models in this package combine an observation with the current state estimate.
"""

__all__ = [
    # submodules
    "probabilistic",
    "linear",
    "kalman",
    # Constants
    "STATE_UPDATERS",
    # Types
    "VectorStateUpdate",
    # Classes
    "CellSequence",
    "ResidualCellSequence",
    "ResidualCell",
    "MissingValueCell",
    "LinearRNNCell",
    "AttentionGain",
    "AttentionCovarianceFactor",
    "InnovationCell",
    "KalmanCell",
    "NonLinearUpdate",
    "NonLinearKalmanUpdate",
    "GradientStepUpdater",
    # probabilistic
    "GaussianForwardUpdater",
    "GaussianReverseUpdater",
    # Imported
    "RNNCell",
    "GRUCell",
    "LSTMCell",
    # Functions
    "is_vector_state_updater",
    "is_consistent_update",
]

from torch.nn import GRUCell, LSTMCell, RNNCell

from . import kalman, linear, probabilistic
from .base import (
    CellSequence,
    MissingValueCell,
    ResidualCell,
    ResidualCellSequence,
    VectorStateUpdate,
    is_consistent_update,
    is_vector_state_updater,
)
from .gradient_based import (
    GradientStepUpdater,
)
from .kalman import NonLinearKalmanUpdate, NonLinearUpdate
from .linear import (
    AttentionCovarianceFactor,
    AttentionGain,
    InnovationCell,
    KalmanCell,
    LinearRNNCell,
)
from .probabilistic import GaussianForwardUpdater, GaussianReverseUpdater

STATE_UPDATERS: dict[str, type[VectorStateUpdate]] = {
    # PyTorch recurrent state updaters
    "GRUCell": GRUCell,
    "LSTMCell": LSTMCell,
    "RNNCell": RNNCell,
    # custom state updaters
    "LinearRNNCell": LinearRNNCell,
    "KalmanCell": KalmanCell,
    "InnovationCell": InnovationCell,
    "MissingValueCell": MissingValueCell,
    "NonLinearKalmanUpdate": NonLinearKalmanUpdate,
    "NonLinearUpdate": NonLinearUpdate,
    "CellSequence": CellSequence,
    "ResidualCell": ResidualCell,
    "ResidualCellSequence": ResidualCellSequence,
}  # fmt: skip
r"""Dictionary of registered state-updater classes."""
