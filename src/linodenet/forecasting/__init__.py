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
]

from linodenet.forecasting.base import (
    PointForecastingModel,
    ProbabilisticForecastingModel,
)
from linodenet.forecasting.linodenet import LinODEnet
from linodenet.forecasting.lssm import LatentStateSpaceModel

FORECASTING_MODELS: dict[str, type[PointForecastingModel]] = {}
r"""Dictionary containing all available forecasting models."""
