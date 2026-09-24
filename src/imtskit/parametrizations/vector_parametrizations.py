r"""Parametrizations for vectors (rank-1 tensors)."""

__all__ = [
    "UnitVector",
    "PositiveVector",
    "StochasticVector",
]

from imtskit.mappings import projections, surjections

UnitVector = projections.UnitVector
StochasticVector = surjections.StochasticVector
PositiveVector = surjections.PositiveVector
