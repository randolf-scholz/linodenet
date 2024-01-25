"""Forecasting Models."""

__all__ = [
    # Constants
    "FORECASTING_MODELS",
    # ABCs & Protocols
    "ForecastingModel",
    "ProbabilisticForecastingModel",
    # Classes
    "LatentStateSpaceModel",
    "LinODEnet",
]

from linodenet.models.forecasting.base import (
    ForecastingModel,
    ProbabilisticForecastingModel,
)
from linodenet.models.forecasting.linodenet import LinODEnet
from linodenet.models.forecasting.lssm import LatentStateSpaceModel

FORECASTING_MODELS: dict[str, type[ForecastingModel]] = {}
"""Dictionary containing all available forecasting models."""
