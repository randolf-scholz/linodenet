"""Manual implementation for the Linear Contraction Layer."""


__all__ = ["LinearContraction"]

from typing import Final

import torch
from torch import Tensor, jit, nn
from torch.nn import functional

from linodenet.lib import singular_triplet


class LinearContraction(nn.Module):
    """Linear layer with a Lipschitz constant."""

    input_size: Final[int]
    """CONST: The input size."""
    output_size: Final[int]
    """CONST: The output size."""
    L: Final[float]
    """CONST: The Lipschitz constant."""
    c: Tensor
    r"""CONST: The regularization hyperparameter."""
    one: Tensor
    r"""CONST: A tensor with value 1.0"""

    weight: Tensor
    """PARAM: The weight matrix."""
    bias: Tensor
    """PARAM: The bias vector."""

    cached_weight: Tensor
    """BUFFER: The cached weight matrix."""
    sigma: Tensor
    """BUFFER: The singular values of the weight matrix."""
    u: Tensor
    """BUFFER: The left singular vectors of the weight matrix."""
    v: Tensor
    """BUFFER: The right singular vectors of the weight matrix."""

    def __init__(
        self, input_size: int, output_size: int, L: float = 1.0, *, bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.L = L

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
        self.register_buffer("c", torch.tensor(self.L))
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
        """Reset the cached weight matrix.

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
        # Note: we need the second run to set up the gradients
        self.recompute_cache()

    @jit.export
    def recompute_cache(self) -> None:
        r"""Recompute the cached weight matrix."""
        # Compute the cached weight matrix
        sigma, u, v = singular_triplet(self.weight, u0=self.u, v0=self.v)
        gamma = torch.minimum(self.one, self.c / sigma)
        cached_weight = gamma * self.weight

        # NOTE: We **MUST** use inplace operations here! Otherwise, we run into the issue that another module
        # that uses weight sharing (e.g. a Linear layer with the same weight matrix) doesn't have the correct *version*
        # of the cached weight matrix when computing backward.
        self.sigma.copy_(sigma)
        self.u.copy_(u)
        self.v.copy_(v)
        self.cached_weight.copy_(cached_weight)  # ✔
        # self.cached_weight = cached_weight  # ✘ (leads to RuntimeError [modified by an inplace operation])

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
            # compute the new parametrized weights from the parameter matrix
            self.recompute_cache()
            # overwrite the parameter with the cached weight matrix
            self.weight.copy_(self.cached_weight)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        # exposed only the cached matrix.
        return functional.linear(x, self.cached_weight, self.bias)
