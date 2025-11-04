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
    "ResidualFilter",
    "ResNetFilter",
    "ReZeroFilter",
    "SequentialFilter",
    "MissingValueFilter",
    "KalmanFilter",
    # Cells
    "LinearCell",
    "LinearKalmanCell",
    "LinearResidualCell",
    "NonLinearCell",
    "NonLinearKalmanCell",
    "PseudoKalmanCell",
    "ResidualCell",
    # Imported
    "RNNCell",
    "GRUCell",
    "LSTMCell",
    # Functions
    "get_filter",
]

from torch.nn import GRUCell, LSTMCell, RNNCell

from linodenet.modules.filters import probabilistic
from linodenet.modules.filters.base import (
    Filter,
    FilterBase,
    ResidualFilter,
    ResNetFilter,
    ReZeroFilter,
    SequentialFilter,
)
from linodenet.modules.filters.cells import (
    Cell,
    CellBase,
    LinearCell,
    LinearKalmanCell,
    LinearResidualCell,
    NonLinearCell,
    NonLinearKalmanCell,
    PseudoKalmanCell,
    ResidualCell,
)
from linodenet.modules.filters.kalman_filter import KalmanFilter
from linodenet.modules.filters.missing_value_filter import MissingValueFilter

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


def get_filter(kind: object = None, /, **cfg: object) -> Filter:
    r"""Initialize from a configuration."""
    match kind:
        # if instance, return as-is
        case Filter() as instance:
            if cfg:
                raise ValueError(f"Cannot pass arguments to an instance: {instance!r}")
            return instance
        # if class, try to instantiate it with the given configuration
        case type() as cls:
            try:
                return cls(**cfg)
            except TypeError as exc:
                exc.add_note(f"Failed to instantiate {cls} with arguments {cfg!r}")
                raise
        # if name, look up in the dictionary
        case str(name):
            _kind = FILTERS[name]
            return get_filter(_kind, **cfg)
        # if config, extract the name and instantiate
        case None:
            if "__module__" in cfg:
                from linodenet.torch_generics import (  # noqa: PLC0415
                    initialize_from_dict,
                )

                result = initialize_from_dict(cfg)
                assert isinstance(result, Filter)
                return result
            try:
                return get_filter(cfg.pop("__name__"), **cfg)
            except KeyError as exc:
                exc.add_note(f"Expected {cfg=} to contain '__name__'")
                raise
        case _:
            raise TypeError(f"Invalid argument: {kind!r}")
