r"""Forecasting Models."""

__all__ = [
    # Constants
    "FORECASTING_MODELS",
    # ABCs & Protocols
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
    "LatentStateSpaceModel",
    "LinODEnet",
    "MarginalizableNormalizingFlow",
    "ProFITi",
    "Shiesh",
]

from .base import PointForecastingModel, ProbabilisticForecastingModel
from .continuous_kalman_filter import ContinuousKalmanFilter
from .cru import CRU
from .discrete_kalman_filter import DiscreteKalmanFilter
from .grafiti import Grafiti
from .gru_d import GRU_D
from .gru_ode_bayes import GRU_ODE_Bayes
from .last_value import LastValue
from .linodenet import LinODEnet
from .lssm import LatentStateSpaceModel
from .mnf import MarginalizableNormalizingFlow
from .profiti import ProFITi, Shiesh

FORECASTING_MODELS: dict[str, type[PointForecastingModel]] = {}
r"""Dictionary containing all available forecasting models."""
