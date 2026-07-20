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
    "ContinuousTimeKalmanFilter",
    "ContinuousTimeNKF",
    "DiscreteTimeKalmanFilter",
    "DiscreteTimeNKF",
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
from .cru import CRU
from .grafiti import Grafiti
from .gru_d import GRU_D
from .gru_ode_bayes import GRU_ODE_Bayes
from .kalman_filter import ContinuousTimeKalmanFilter, DiscreteTimeKalmanFilter
from .last_value import LastValue
from .mnf import Moses
from .neural_flow import NeuralFlow
from .normalizing_kalman_filter import ContinuousTimeNKF, DiscreteTimeNKF
from .profiti import ProFITi

# TODO: allow time marginal gaussian models to be treated as point predictiors.
POINT_FORECASTING_MODELS: dict[str, type[PointForecastingModel]] = {
    "Grafiti": Grafiti,
    "GRU_D": GRU_D,
    "LastValue": LastValue,
}
r"""Dictionary containing all available forecasting models."""

PROBABILISTIC_FORECASTING_MODELS: dict[str, type[ProbabilisticForecastingModel]] = {
    "ContinuousTimeKalmanFilter": ContinuousTimeKalmanFilter,
    "ContinuousTimeNKF": ContinuousTimeNKF,
    "CRU": CRU,
    "GRU_ODE_Bayes": GRU_ODE_Bayes,
    "NeuralFlow": NeuralFlow,
    "DiscreteTimeNKF": DiscreteTimeNKF,
}
r"""Dictionary containing all available probabilistic forecasting models."""

PATH_FORECASTING_MODELS: dict[str, type[PathForecastingModel]] = {
    "ProFITi": ProFITi,
    "Moses": Moses,
}
r"""Dictionary containing all available path forecasting models."""
