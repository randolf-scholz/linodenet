"""Implementation of invertible layers.

Layers:
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
    "NaiveLinearContraction",
    "LinearContraction",
    "iResNetBlock",
    "iSequential",
]

import warnings
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn
from torch.nn import functional
from typing_extensions import Self

from linodenet.lib import singular_triplet


class NaiveLinearContraction(nn.Module):
    """Linear layer with a Lipschitz constant.

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

    def forward(self, x):
        sigma = torch.linalg.matrix_norm(self.weight, ord=2)
        gamma = torch.minimum(self.c / sigma, self.one)
        return functional.linear(x, gamma * self.weight, self.bias)
        # return self.layer(x / sigma)


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
            bound = float(torch.rsqrt(torch.tensor(self.input_size)))
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
            self.recompute_cache()
            self.weight.copy_(self.cached_weight)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        return functional.linear(x, self.cached_weight, self.bias)


class iResNetBlock(nn.Module):
    """A single block of an iResNet.

    References:
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | http://proceedings.mlr.press/v97/behrmann19a.html
    """

    maxiter: Final[int]
    r"""CONST: Maximum number of steps in power-iteration"""
    atol: Final[float]
    r"""CONST: Absolute tolerance for fixed point iteration"""
    rtol: Final[float]
    r"""CONST: Relative tolerance for fixed point iteration"""
    is_inverse: Final[bool]
    r"""CONST: Whether to use the inverse or the forward map"""

    converged: Tensor
    r"""BUFFER: Boolean tensor indicating convergence"""

    def __init__(
        self,
        layer: nn.Module,
        *,
        maxiter: int = 100,
        atol: float = 1e-5,
        rtol: float = 1e-4,
        inverse: Optional[Self] = None,
    ) -> None:
        super().__init__()
        self.block = layer
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.register_buffer("converged", torch.tensor(False))

        self.is_inverse = inverse is not None
        if inverse is None:
            cls = type(self)
            self.inverse = cls(
                layer,
                maxiter=maxiter,
                atol=atol,
                rtol=rtol,
                inverse=self,
            )

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Compute the forward map with residual connection."""
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or tolerance threshold
        $|x'-x|≤\text{atol} + \text{rtol}⋅|x|$ is reached.
        """
        if self.is_inverse:
            return self._encode(y)
        return self._decode(y)

    @jit.export
    def _encode(self, x: Tensor) -> Tensor:
        return x + self.block(x)

    @jit.export
    def _decode(self, y: Tensor) -> Tensor:
        x = y.clone().detach()
        # m = torch.isnan(y)
        residual = torch.zeros_like(y)

        for _ in range(self.maxiter):
            x_prev = x
            x = y - self.block(x)
            with torch.no_grad():
                residual = torch.sqrt(torch.nansum((x - x_prev).pow(2)))
                tol = self.atol + self.rtol * torch.sqrt(torch.nansum(x_prev.pow(2)))
                self.converged = residual < tol
                if self.converged:
                    break
        if not self.converged:
            warnings.warn(
                f"No convergence in {self.maxiter} iterations. "
                f"Max residual:{residual.item()} > {self.atol}.",
                stacklevel=2,
            )
        return x


class iSequential(nn.Module):
    r"""Invertible Sequential model."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    is_inverse: Final[bool]
    r"""CONST: Whether to use the inverse or the forward map"""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "maxiter": 10,
        "input_size": None,
        "dropout": None,
        "bias": True,
        "nblocks": 5,
        "rezero": False,
        "iResNetBlock": {
            "input_size": None,
            "activation": "ReLU",
            "activation_config": {"inplace": False},
            "bias": True,
            "hidden_size": None,
            "maxiter": 100,
        },
    }
    r"""The hyperparameter dictionary"""

    @classmethod
    def from_config(cls, **cfg: Any) -> Self:
        raise NotImplementedError

    def __new__(
        cls, *modules: nn.Module, inverse: Optional[Self] = None, **cfg: Any
    ) -> Self:
        r"""Initialize from hyperparameters."""
        blocks: list[nn.Module] = [] if modules is None else list(modules)
        assert len(blocks) ^ len(cfg), "Provide either blocks, or hyperparameters!"

        if cfg:
            return cls.from_config(**cfg)

        return super().__new__(cls)

    def __init__(self, *modules: nn.Module, inverse: Optional[Self] = None) -> None:
        r"""Initialize from hyperparameters."""
        super().__init__()

        blocks: list[nn.Module] = [] if modules is None else list(modules)
        # assert len(blocks) ^ len(hparams), "Provide either blocks, or hyperparameters!"
        # if hparams:
        #     raise ValueError

        self.input_size = -1
        self.output_size = -1

        # self.input_size = blocks[0].input_size  # type: ignore[assignment]
        # self.output_size = blocks[-1].output_size  # type: ignore[assignment]
        self.blocks = nn.Sequential(*blocks)

        # print([layer.is_inverse for layer in self])

        self.is_inverse = inverse is not None
        if not self.is_inverse:
            cls = type(self)
            self.inverse = cls(
                *[block.inverse for block in reversed(self.blocks)], inverse=self
            )

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """Compute the encoding."""
        return self.blocks(x)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        """Compute the encoding."""
        return self.blocks(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fix point iteration in each block in reversed order."""
        for layer in self.blocks[::-1]:  # traverse in reverse
            y = layer.decode(y)
        return y
