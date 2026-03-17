r"""Diffeomorphisms, i.e. differentiable bijections with differentiable inverse.

We call a module a bijection if it satisfies 3 properties:

1. It has both an `encode` and `decode` method.
2. It is invertible, i.e. `decode(encode(x)) = x` and `encode(decode(y)) = y`
3. Both encode and decode are differentiable.

Examples:
    - Affine: $y = Ax+b$ and $x = A⁻¹(y-b)$
        - A diagonal
        - A triangular
        - A tridiagonal
    - Element-wise:
        - Monotonic
    - Shears (coupling flows): $y_A = f(x_A, x_B)$ and $y_B = x_B$
        - Example: $y_A = x_A + e^{x_B}$ and $y_B = x_B$
    - Residual: $y = x + F(x)$
        - Contractive: $y = F(x)$ with $‖F‖<1$
        - Low Rank Perturbation: $y = x + ABx$
    - Continuous Time Flows: $ẋ=f(t, x)$
"""

__all__ = [
    # constants
    "BIJECTIONS",
    # protocols & base classes
    "InverseTransform",
    "Transform",
    "TransformBase",
    "TransformSequence",
    # classes
    "iResNet",
    "iResNetBlock",
    "iResNetLayer",
    "LowRankFlow",
    "SplineFlow",
    "TriangularFlow",
    "ContractiveFlow",
]


from linodenet.flows.transforms.base import (
    InverseTransform,
    Transform,
    TransformBase,
    TransformSequence,
)
from linodenet.flows.transforms.contractive import ContractiveFlow
from linodenet.flows.transforms.iresnet import iResNet, iResNetBlock, iResNetLayer
from linodenet.flows.transforms.low_rank import LowRankFlow
from linodenet.flows.transforms.spline_flow import SplineFlow
from linodenet.flows.transforms.triangular import TriangularFlow
from linodenet.mappings.base import (
    Bijection,
    BijectionBase,
    BijectionSequence,
    InverseBijection,
)

BIJECTIONS: dict[str, type[Bijection]] = {
    "iResNet": iResNet,
    "SplineFlow": SplineFlow,
    "LowRankFlow": LowRankFlow,
    "TriangularFlow": TriangularFlow,
}  # fmt: skip
r"""Dictionary containing all available bijections."""
