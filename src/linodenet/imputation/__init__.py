r"""Imputers for missing data handling."""

__all__ = [
    # constants
    "IMPUTERS",
    # types
    "ImputerProtocol",
    "ImputationStrategy",
    # classes
    "ZeroImputer",
    "ConstantImputer",
    "LastValueImputer",
    "LinearImputer",
    # functions
    "zero_impute",
]

from .base import (
    ConstantImputer,
    ImputationStrategy,
    ImputerProtocol,
    LastValueImputer,
    LinearImputer,
    ZeroImputer,
    zero_impute,
)

IMPUTERS: dict[str, type[ImputerProtocol]] = {
    "ZeroImputer"      : ZeroImputer,
    "ConstantImputer"  : ConstantImputer,
    "LastValueImputer" : LastValueImputer,
    "LinearImputer"    : LinearImputer,
}  # fmt: skip
r"""Dictionary of available imputers."""
