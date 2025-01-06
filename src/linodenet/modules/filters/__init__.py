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
    # ABCs & Protocols
    "Cell",
    "CellBase",
    "Filter",
    "FilterBase",
    # Classes
    "MissingValueFilter",
    "ReZeroFilter",
    "ResNetFilter",
    "ResidualFilter",
    "SequentialFilter",
    # Functions
    "get_filter",
]

from torch.nn import GRUCell, LSTMCell, RNNCell

from linodenet.modules.filters import probabilistic
from linodenet.modules.filters.base import (
    Filter,
    FilterBase,
    MissingValueFilter,
    ResidualFilter,
    ResNetFilter,
    ReZeroFilter,
    SequentialFilter,
)
from linodenet.modules.filters.cells import (
    Cell,
    CellBase,
    LinearCell,
    NonLinearCell,
    NonLinearKalmanCell,
)
from linodenet.modules.filters.filters import (
    KalmanFilter,
    LinearKalmanCell,
    LinearResidualCell,
    PseudoKalmanCell,
    ResidualCell,
)

x: type[Cell] = LinearCell

CELLS: dict[str, type[Cell]]  = {
    # torch cells
    "GRUCell"            : GRUCell,
    "LSTMCell"           : LSTMCell,
    "RNNCell"            : RNNCell,
    # custom cells
    "KalmanCell"         : NonLinearKalmanCell,
    "LinearCell"         : LinearCell,
    "LinearKalmanCell"   : LinearKalmanCell,
    "LinearResidualCell" : LinearResidualCell,
    # "MissingValueCell"   : MissingValueCell,
    "NonLinearCell"      : NonLinearCell,
    "PseudoKalmanCell"   : PseudoKalmanCell,
    "ResidualCell"       : ResidualCell,
}  # fmt: skip
r"""Dictionary of all available cells (basic building blocks for filters)."""

FILTERS: dict[str, type[Filter]] = {
    "KalmanFilter"        : KalmanFilter,
    "MissingValueFilter"  : MissingValueFilter,
    # "ProbabilisticFilter" : ProbabilisticFilter,
    "ResidualFilter"      : ResidualFilter,
    # "ResidualFilterBlock" : ResidualFilterBlock,
    "SequentialFilter"    : SequentialFilter,
}  # fmt: skip
r"""Dictionary of all available filters."""


def get_filter(filter_kind: str | type | None = None, /, **cfg: object) -> Filter:
    r"""Initialize from a configuration."""
    match filter_kind:
        case None:
            filter_name = cfg.pop("name")
            return get_filter(filter_name, **cfg)  # type: ignore[arg-type]
        case type() as cls:
            try:
                return cls(**cfg)
            except Exception as exc:
                raise RuntimeError(f"Failed to create filter of type {cls}!") from exc
        case str(name):
            typ: type = FILTERS[name]
            return get_filter(typ, **cfg)
        case _:
            raise TypeError(f"Invalid argument type: {filter_kind!r}")
