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
]

from linodenet.forecasting.base import (
    PointForecastingModel,
    ProbabilisticForecastingModel,
)
from linodenet.forecasting.continuous_kalman_filter import ContinuousKalmanFilter
from linodenet.forecasting.discrete_kalman_filter import DiscreteKalmanFilter
from linodenet.forecasting.linodenet import LinODEnet
from linodenet.forecasting.lssm import LatentStateSpaceModel

FORECASTING_MODELS: dict[str, type[PointForecastingModel]] = {}
r"""Dictionary containing all available forecasting models."""
