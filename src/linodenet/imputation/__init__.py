r"""Imputers for missing data handling."""

__all__ = [
    # constants
    "IMPUTERS",
    # types
    "ImputerProtocol",
    "ImputationStrategy",
    # classes
    "ZeroImputer",
    "ConstantValueImputer",
    "LearnableValueImputer",
    "LastValueImputer",
    "LinearImputer",
    # functions
    "zero_impute",
]

from linodenet.imputation.base import (
    ConstantValueImputer,
    ImputationStrategy,
    ImputerProtocol,
    LastValueImputer,
    LearnableValueImputer,
    LinearImputer,
    ZeroImputer,
    zero_impute,
)

IMPUTERS: dict[str, type[ImputerProtocol]] = {
    "ZeroImputer"           : ZeroImputer,
    "ConstantValueImputer"  : ConstantValueImputer,
    "LearnableValueImputer" : LearnableValueImputer,
    "LastValueImputer"      : LastValueImputer,
    "LinearImputer"         : LinearImputer,
}  # fmt: skip
r"""Dictionary of available imputers."""
