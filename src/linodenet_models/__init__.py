r"""Forecasting Models."""

__all__ = [
    # Constants
    "PATH_FORECASTING_MODELS",
    "POINT_FORECASTING_MODELS",
    "PROBABILISTIC_FORECASTING_MODELS",
    # ABCs & Protocols
    "PathForecastingModel",
    "PointForecastingModel",
    "ProbabilisticForecastingModel",
    # Classes
    "CRU",
    "ContinuousKalmanFilter",
    "DiscreteKalmanFilter",
    "GRU_D",
    "GRU_ODE_Bayes",
    "Grafiti",
    "LastValue",
    "Moses",
    "NeuralFlow",
    "ProFITi",
]


from .base import (
    PathForecastingModel,
    PointForecastingModel,
    ProbabilisticForecastingModel,
)
from .continuous_kalman_filter import ContinuousKalmanFilter
from .cru import CRU
from .discrete_kalman_filter import DiscreteKalmanFilter
from .grafiti import Grafiti
from .gru_d import GRU_D
from .gru_ode_bayes import GRU_ODE_Bayes
from .last_value import LastValue
from .mnf import Moses
from .neural_flow import NeuralFlow
from .profiti import ProFITi

# TODO: allow time marginal gaussian models to be treated as point predictiors.
POINT_FORECASTING_MODELS: dict[str, type[PointForecastingModel]] = {
    "grafiti": Grafiti,
    "gru_d": GRU_D,
    "last_value": LastValue,
}
r"""Dictionary containing all available forecasting models."""

PROBABILISTIC_FORECASTING_MODELS: dict[str, type[ProbabilisticForecastingModel]] = {
    "continuous_kalman_filter": ContinuousKalmanFilter,
    "cru": CRU,
    "gru_ode_bayes": GRU_ODE_Bayes,
    "neural_flow": NeuralFlow,
}
r"""Dictionary containing all available probabilistic forecasting models."""

PATH_FORECASTING_MODELS: dict[str, type[PathForecastingModel]] = {
    "profiti": ProFITi,
    "moses": Moses,
}
r"""Dictionary containing all available path forecasting models."""
