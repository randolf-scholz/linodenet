r"""Forecasting Models."""

__all__ = [
    # Constants
    "FORECASTING_MODELS",
    # ABCs & Protocols
    "PointForecastingModel",
    "ProbabilisticForecastingModel",
    # Classes
    "LatentStateSpaceModel",
    "LinODEnet",
    "ContinuousKalmanFilter",
    "DiscreteKalmanFilter",
    "LastValue",
]

from .base import PointForecastingModel, ProbabilisticForecastingModel
from .continuous_kalman_filter import ContinuousKalmanFilter
from .discrete_kalman_filter import DiscreteKalmanFilter
from .last_value import LastValue
from .linodenet import LinODEnet
from .lssm import LatentStateSpaceModel

FORECASTING_MODELS: dict[str, type[PointForecastingModel]] = {}
r"""Dictionary containing all available forecasting models."""
