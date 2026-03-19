r"""Re-Implementation of Spectral Normalization Layer.

Notes:
    The default implementation provided by torch is problematic [1]_:

    - It only performs a single power iteration, without any convergence test.
    - It internally uses pseudo-inverse, which causes a full SVD to be computed.
        - This SVD can fail for ill-conditioned matrices.
    - It does not implement any optimized backward pass.

    Alternatively, torch.linalg.matrix_norm(A, ord=2) can be used to compute the spectral norm of a matrix A.
    But here, again, torch computes the full SVD.

    Our implementation addresses these issues:

    - We use the analytic formula for the gradient: $∂‖A‖₂/∂A = uvᵀ$,
      where $u$ and $v$ are the left and right singular vectors of $A$.
    - We use the power iteration with convergence test.


References:
    .. [1] https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm
"""

__all__ = ["LinearContraction"]

from typing import Final, override

from torch import nn

from linodenet.mappings.projections import LipschitzBounded
from linodenet.nn.parametrize import register_parametrization


class LinearContraction(nn.Linear):
    r"""A linear layer $f(x) = A⋅x$ satisfying the contraction property $‖f(x)-f(y)‖₂ ≤ ‖x-y‖₂$.

    This is achieved by normalizing the weight matrix by
    $A' = A⋅\min(\tfrac{c}{‖A‖₂}, 1)$, where $c<1$ is a hyperparameter.
    """

    input_size: Final[int]
    output_size: Final[int]

    @override
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: object = None,
        dtype: object = None,
        c: float = 0.97,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.input_size = self.in_features
        self.output_size = self.out_features

        register_parametrization(self, "weight", LipschitzBounded(lipschitz_bound=c))
