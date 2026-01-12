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

__all__ = [
    "LinearContraction",
    "AltLinearContraction",
    "NaiveLinearContraction",
    "LinearContractionManualParametrized",
]


from math import sqrt
from typing import Final, Optional

import torch
from torch import Tensor, jit, nn
from torch.linalg import matrix_norm
from torch.nn import functional

from linodenet.lib import singular_triplet
from linodenet.signatures import signature


class LinearContraction(nn.Module):
    r"""A linear layer $f(x) = A⋅x$ satisfying the contraction property $‖f(x)-f(y)‖_2 ≤ ‖x-y‖_2$.

    This is achieved by normalizing the weight matrix by
    $A' = A⋅\min(\tfrac{c}{‖A‖_2}, 1)$, where $c<1$ is a hyperparameter.

    Attributes:
        input_size:  int
            The dimensionality of the input space.
        output_size: int
            The dimensionality of the output space.
        c: Tensor
            The regularization hyperparameter.
        spectral_norm: Tensor
            BUFFER: The value of `‖W‖_2`
        weight: Tensor
            The weight matrix.
        bias: Tensor or None
            The bias Tensor if present, else None.
    """

    input_size: Final[int]
    output_size: Final[int]

    # Constants
    c: Tensor
    r"""CONST: The regularization hyperparameter."""
    one: Tensor
    r"""CONST: A tensor with value 1.0"""

    # Buffers
    spectral_norm: Tensor
    r"""BUFFER: The value of $‖W‖_2$"""

    # Parameters
    weight: Tensor
    r"""PARAM: The weight matrix."""
    bias: Optional[Tensor]
    r"""PARAM: The bias term."""

    def __init__(
        self, input_size: int, output_size: int, *, c: float = 0.97, bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(Tensor(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(Tensor(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.register_buffer("one", torch.tensor(1.0), persistent=True)
        self.register_buffer("c", torch.tensor(float(c)), persistent=True)
        self.register_buffer(
            "spectral_norm", matrix_norm(self.weight, ord=2), persistent=False
        )

    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            bound = 1 / sqrt(self.input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    # def extra_repr(self) -> str:
    #     return "input_size={}, output_size={}, bias={}".format(
    #         self.input_size, self.output_size, self.bias is not None
    #     )

    @jit.export
    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        # σ_max, _ = torch.lobpcg(self.weight.T @ self.weight, largest=True)
        # σ_max = torch.linalg.norm(self.weight, ord=2)
        # self.spectral_norm = spectral_norm(self.weight)
        # σ_max = torch.linalg.svdvals(self.weight)[0]
        self.spectral_norm = matrix_norm(self.weight, ord=2)
        gamma = torch.minimum(self.c / self.spectral_norm, self.one)
        return functional.linear(x, gamma * self.weight, self.bias)


class AltLinearContraction(nn.Module):
    r"""A linear layer `f(x) = A⋅x` satisfying the contraction property `‖f(x)-f(y)‖_2 ≤ ‖x-y‖_2`.

    This is achieved by normalizing the weight matrix by
    `A' = A⋅\min(\tfrac{c}{‖A‖_2}, 1)`, where `c<1` is a hyperparameter.

    Attributes:
        input_size:  int
            The dimensionality of the input space.
        output_size: int
            The dimensionality of the output space.
        c: Tensor
            The regularization hyperparameter
        kernel: Tensor
            The weight matrix
        bias: Tensor or None
            The bias Tensor if present, else None.
    """

    # Constants
    input_size: Final[int]
    r"""CONST:  Number of inputs"""
    output_size: Final[int]
    r"""CONST: Number of outputs"""
    maxiter: Final[int]
    r"""CONST: Maximum number of steps in power-iteration"""

    # Buffers
    c: Tensor
    r"""BUFFER: The regularization strength."""
    one: Tensor
    r"""BUFFER: Constant value of float(1.0)."""
    spectral_norm: Tensor
    r"""BUFFER: The largest singular value."""
    u: Tensor
    r"""BUFFER: The left singular vector."""
    v: Tensor
    r"""BUFFER: The right singular vector."""

    # Parameters
    kernel: Tensor
    r"""PARAM: the weight matrix"""
    bias: Optional[Tensor]
    r"""PARAM: The bias term"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        c: float = 0.97,
        bias: bool = True,
        maxiter: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.maxiter = maxiter

        self.kernel = nn.Parameter(Tensor(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(Tensor(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # self.spectral_norm = matrix_norm(self.weight, ord=2)
        self.register_buffer("one", torch.tensor(1.0))
        self.register_buffer("c", torch.tensor(float(c)))
        self.register_buffer("spectral_norm", matrix_norm(self.kernel, ord=2))
        # self.register_buffer(
        #     "u",
        # )
        # self.register_buffer(
        #     "v",
        # )

    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        nn.init.kaiming_uniform_(self.kernel, a=sqrt(5))
        if self.bias is not None:
            bound = 1 / sqrt(self.input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    # def extra_repr(self) -> str:
    #     return "input_size={}, output_size={}, bias={}".format(
    #         self.input_size, self.output_size, self.bias is not None
    #     )

    @jit.export
    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        # σ_max, _ = torch.lobpcg(self.weight.T @ self.weight, largest=True)
        # σ_max = torch.linalg.norm(self.weight, ord=2)
        # σ_max = spectral_norm(self.weight)
        # σ_max = torch.linalg.svdvals(self.weight)[0]
        self.spectral_norm = matrix_norm(self.kernel, ord=2)
        fac = torch.minimum(self.c / self.spectral_norm, self.one)
        return functional.linear(x, fac * self.kernel, self.bias)


class NaiveLinearContraction(nn.Module):
    r"""Linear layer with a Lipschitz constant.

    Note:
        Naive implementation using the builtin matrix_norm function.
        This is very slow and should only be used for testing.
        The backward is unstable and will often fail.
    """

    # Parameters
    weight: Tensor
    r"""PARAM: The weight matrix."""
    bias: Optional[Tensor]
    r"""PARAM: The bias term."""
    one: Tensor
    c: Tensor

    def __init__(
        self, input_size: int, output_size: int, *, c: float = 1.0, bias: bool = False
    ):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size, bias=bias)
        self.weight = self.layer.weight
        self.bias = self.layer.bias
        self.register_buffer("c", torch.tensor(float(c)), persistent=True)
        self.register_buffer("one", torch.tensor(1.0), persistent=True)

    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        sigma = matrix_norm(self.weight, ord=2)
        gamma = torch.minimum(self.c / sigma, self.one)
        return functional.linear(x, gamma * self.weight, self.bias)


class LinearContractionManualParametrized(nn.Module):
    r"""Linear layer with a Lipschitz constant."""

    input_size: Final[int]
    r"""CONST: The input size."""
    output_size: Final[int]
    r"""CONST: The output size."""
    lipschitz_constant: Final[float]
    r"""CONST: The Lipschitz constant."""
    c: Tensor
    r"""CONST: The regularization hyperparameter."""
    one: Tensor
    r"""CONST: A tensor with value 1.0"""

    weight: Tensor
    r"""PARAM: The weight matrix."""
    bias: Tensor
    r"""PARAM: The bias vector."""

    cached_weight: Tensor
    r"""BUFFER: The cached weight matrix."""
    sigma: Tensor
    r"""BUFFER: The singular values of the weight matrix."""
    u: Tensor
    r"""BUFFER: The left singular vectors of the weight matrix."""
    v: Tensor
    r"""BUFFER: The right singular vectors of the weight matrix."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        lipschitz_constant: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lipschitz_constant = lipschitz_constant

        self.weight = nn.Parameter(torch.empty((output_size, input_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.register_buffer("sigma", torch.tensor(1.0))
        self.register_buffer("u", torch.randn(output_size))
        self.register_buffer("v", torch.randn(input_size))
        self.register_buffer("cached_weight", torch.empty_like(self.weight))
        self.register_buffer("one", torch.ones(1))
        self.register_buffer("c", torch.tensor(self.lipschitz_constant))
        self.reset_cache()

    @jit.export
    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        with torch.no_grad():
            bound: float = float(torch.rsqrt(torch.tensor(self.input_size)))
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    @jit.export
    def reset_cache(self) -> None:
        r"""Reset the cached weight matrix.

        Needs to be called after every .backward!
        """
        # apply projection step.
        self.projection()

        # detach() is necessary to avoid "Trying to backward through the graph a second time" error
        self.sigma.detach_()
        self.u.detach_()
        self.v.detach_()
        self.cached_weight.detach_()

        # recompute the cache
        # NOTE: we need the second run to set up the gradients
        self.recompute_cache()

    @jit.export
    def recompute_cache(self) -> None:
        r"""Recompute the cached weight matrix."""
        # Compute the cached weight matrix
        sigma, u, v = singular_triplet(self.weight, u0=self.u, v0=self.v)
        gamma = torch.minimum(self.one, self.c / sigma)
        cached_weight = gamma * self.weight

        # NOTE: We MUST use inplace operations here! Otherwise, we run into the issue that
        #  when another module uses weight sharing (e.g. a Linear layer with the same weight matrix)
        #  doesn't have the correct version of the cached weight matrix when computing backward.
        self.sigma.copy_(sigma)
        self.u.copy_(u)
        self.v.copy_(v)
        self.cached_weight.copy_(cached_weight)  # ✅️
        # self.cached_weight = cached_weight  # ❌️ (leads to RuntimeError [modified by an inplace operation])

    @jit.export
    def projection(self) -> None:
        r"""Project the cached weight matrix.

        NOTE: regarding the order of operations, we ought to use projected gradient descent:

        .. math:: w' = proj(w - γ ∇w L(w/‖w‖))

        So the order should be:

        1. compute $w̃ = w/‖w‖₂$
        2. compute $∇w L(w̃)$
        3. apply the gradient descent step $w' = w - γ ∇w L(w̃)$
        4. project $w' ← w'/‖w'‖₂$

        Now, this is a bit ineffective since it requires us to compute ‖w‖ twice.
        But, that's the price we pay for the projection step.

        Note:
            - The following is not quite correct, since we substitute the spectral norm with the euclidean norm.
            - even if ‖w‖=1, we still compute this step to get gradient information.
            - since ∇w w/‖w‖ = 𝕀/‖w‖ - ww^⊤/‖w‖³ = (𝕀 - ww^⊤), then for outer gradient ξ,
              the VJP is given by ξ - (ξ^⊤w)w which is the projection of ξ onto the tangent space.

        NOTE: Riemannian optimization on n-sphere:
        Given point p on unit sphere and tangent vector v, the geodesic is given by:

        .. math:: γ(t) = cos(‖v‖t)p + sin(‖v‖t)v/‖v‖

        t takes the role of the step size. For small t, we can use the second order approximation:

        .. math:: γ(t) ≈ p + t v - ½t²‖v‖²p

        Which is a great approximation for small t. The projection still needs to be applied.
        """
        with torch.no_grad():
            self.recompute_cache()
            self.weight.copy_(self.cached_weight)

    @jit.export
    @signature("(..., n) -> (..., n)")
    def forward(self, x: Tensor) -> Tensor:
        return functional.linear(x, self.cached_weight, self.bias)
