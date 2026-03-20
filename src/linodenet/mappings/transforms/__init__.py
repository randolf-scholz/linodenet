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


Protocols and base classes for bijections and transforms.

Note that `torch.distributions.Transform` has some differences:

- It is not a protocol.
- `log_abs_det_jacobian` always requires 2 arguments, $x$ and $y$.
  The rationale is that for certain bijections, the Jacobian is
  much faster to compute if the output is known.
  For example: if $f(x) = xᵃ$, then $\log\abs{\det 𝐃f[x]} = \log\abs{a⋅y/x}$.
  is more efficient than $\log\abs{a⋅xᵃ⁻¹}$.
  However, for many bijections, this is not true and knowing $y$ is not that helpful.
- Instead, it makes more sense to have 2 methods: `log_abs_det_jacobian(x)`
  and `value_and_log_abs_det_jacobian(x) -> tuple[Tensor, Tensor]`,
  similar to jax's `value_and_grad`.
  Alternatively, one can store `y` in a buffer and reuse it if needed, i.e.
  methods that need `y` can call:

>>>
>>> def log_abs_det_jacobian(self, x: Tensor, /, y: None | Tensor = None) -> Tensor:
>>>     if y is None:
>>>         if id(x) == id(self._last_x):
>>>             y = self._last_y
>>>         else:
>>>             y = self.encode(x)
"""

__all__ = [
    # constants
    # protocols & base classes
    # classes
    "iResNet",
    "iResNetBlock",
    "iResNetLayer",
    "LowRankTransform",
    "SplineTransform",
    "TriangularTransform",
    "ResidualContraction",
    "ReZeroContraction",
    "ResidualContractionFallback",
]

from linodenet.mappings.transforms.iresnet import iResNet, iResNetBlock, iResNetLayer
from linodenet.mappings.transforms.linear_rational_spline import SplineTransform
from linodenet.mappings.transforms.low_rank import LowRankTransform
from linodenet.mappings.transforms.residual import (
    ResidualContraction,
    ResidualContractionFallback,
    ReZeroContraction,
)
from linodenet.mappings.transforms.triangular import TriangularTransform
