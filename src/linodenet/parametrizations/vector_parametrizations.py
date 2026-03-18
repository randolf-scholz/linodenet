r"""Parametrizations for vectors (rank-1 tensors)."""

__all__ = [
    "UnitVector",
    "NonNegativeVector",
    "SimplexVector",
]

from linodenet.nn.parametrize import WrappedParametrization


class UnitVector(WrappedParametrization):
    r"""Parametrization that constrains a vector to have unit norm.

    Args:
        tensor (Tensor): The tensor to be parametrized.
    """


class NonNegativeVector(WrappedParametrization):
    r"""Parametrization that constrains a vector to be non-negative.

    Args:
        tensor (Tensor): The tensor to be parametrized.
    """


class SimplexVector(WrappedParametrization):
    r"""Parametrization that constrains a vector to lie on the simplex.

    Args:
        tensor (Tensor): The tensor to be parametrized.
    """
