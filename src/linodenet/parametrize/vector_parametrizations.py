r"""Parametrizations for vectors (rank-1 tensors)."""

__all__ = [
    "UnitVector",
    "NonNegativeVector",
    "SimplexVector",
]

from linodenet.parametrize.base import Parametrized


class UnitVector(Parametrized):
    r"""Parametrization that constrains a vector to have unit norm.

    Args:
        tensor (Tensor): The tensor to be parametrized.
    """


class NonNegativeVector(Parametrized):
    r"""Parametrization that constrains a vector to be non-negative.

    Args:
        tensor (Tensor): The tensor to be parametrized.
    """


class SimplexVector(Parametrized):
    r"""Parametrization that constrains a vector to lie on the simplex.

    Args:
        tensor (Tensor): The tensor to be parametrized.
    """
