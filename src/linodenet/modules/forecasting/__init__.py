r"""Forecasting Models."""

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

from linodenet.modules.forecasting.base import (
    ForecastingModel,
    ProbabilisticForecastingModel,
)
from linodenet.modules.forecasting.linodenet import LinODEnet
from linodenet.modules.forecasting.lssm import LatentStateSpaceModel

FORECASTING_MODELS: dict[str, type[ForecastingModel]] = {}
r"""Dictionary containing all available forecasting models."""
