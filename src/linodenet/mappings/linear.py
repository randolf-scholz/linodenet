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

__all__ = ["LinearContraction", "RankOneContraction"]

import math
from typing import Final, override

import torch
from torch import Tensor, nn

from linodenet.mappings.projections import Contraction, UnitVector
from linodenet.nn.parametrize import register_parametrization, update_parametrizations


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
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        c: float = 0.97,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.input_size = self.in_features
        self.output_size = self.out_features

        register_parametrization(self, "weight", Contraction(lipschitz_bound=c))
        self.parametrizations: nn.ModuleDict
        assert isinstance(self.parametrizations.weight, nn.Module)
        param = self.parametrizations.weight.original_parameter
        assert isinstance(param, nn.Parameter)
        self.weight_parameter: nn.Parameter = param


class RankOneContraction(nn.Module):
    r"""A rank-1 linear layer $f(x) = c⋅u(vᵀx) + b$ with contraction constant at most $c$.

    For a rank-1 matrix $A = uvᵀ$, the spectral norm satisfies $‖A‖₂ = ‖u‖₂ ‖v‖₂$.
    By parametrizing both factors as unit vectors, the resulting weight
    $A = c⋅uvᵀ$ has spectral norm exactly $c$.
    """

    in_features: Final[int]
    out_features: Final[int]
    input_size: Final[int]
    output_size: Final[int]
    gamma: Tensor
    weight_u: Tensor
    weight_v: Tensor
    bias: nn.Parameter | None

    @override
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        c: float = 0.97,
    ) -> None:
        super().__init__()
        if not 0 < c < 1:
            raise ValueError("c must be between 0 and 1")

        self.in_features = in_features
        self.out_features = out_features
        self.input_size = in_features
        self.output_size = out_features
        self.register_buffer(
            "gamma", torch.tensor(float(c), device=device, dtype=dtype)
        )

        self.weight_u = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype)
        )
        self.weight_v = nn.Parameter(
            torch.empty(in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        register_parametrization(self, "weight_u", UnitVector())
        register_parametrization(self, "weight_v", UnitVector())
        self.parametrizations: nn.ModuleDict
        assert isinstance(self.parametrizations.weight_u, nn.Module)
        assert isinstance(self.parametrizations.weight_v, nn.Module)
        u_param = self.parametrizations.weight_u.original_parameter
        v_param = self.parametrizations.weight_v.original_parameter
        assert isinstance(u_param, nn.Parameter)
        assert isinstance(v_param, nn.Parameter)
        self.weight_u_parameter: nn.Parameter = u_param
        self.weight_v_parameter: nn.Parameter = v_param

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_u.unsqueeze(-1), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_v.unsqueeze(0), a=math.sqrt(5))
        if self.bias is None:
            return

        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)
        update_parametrizations(self)

    @property
    def weight(self) -> Tensor:
        return torch.outer(self.gamma * self.weight_u, self.weight_v)

    def forward(self, x: Tensor) -> Tensor:
        projection = x.matmul(self.weight_v)
        y = torch.einsum("..., o -> ...o", projection, self.gamma * self.weight_u)
        if self.bias is not None:
            y = y + self.bias
        return y
