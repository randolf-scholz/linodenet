r"""Checks for testing certain module properties."""

__all__ = [
    # ABCs & Protocols
    "ModuleTest",
    # Functions
    "assert_backward_stable",
    "assert_forward_stable",
    "get_output",
    "is_backward_stable",
    "is_forward_stable",
]

from collections.abc import Callable
from typing import Optional, Protocol

import torch
from torch import Tensor, nn

from linodenet.constants import ATOL, RTOL
from linodenet.testing.statistics import is_standardized


class ModuleTest(Protocol):
    r"""Protocol for Module Testing."""

    def __call__(
        self,
        module: nn.Module,
        /,
        *,
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> bool:
        r"""Test the module."""
        ...


def get_output(func: Callable[..., Tensor], /, *inputs: Tensor) -> Tensor:
    batch_size = inputs[0].shape[0]
    assert all(x.shape[0] == batch_size for x in inputs)

    # run the forward pass
    try:
        output = func(*inputs)
    except Exception as exc:
        exc.add_note(f"Error in forward pass of {func}")
        raise

    # make sure the output is valid
    if not isinstance(output, Tensor):
        raise TypeError(f"Expected a tensor, but got {type(output)}")

    if output.ndim <= 1 or output.shape[0] != batch_size:
        raise ValueError(f"Expected a batched output, but got {output.shape}")

    if not output.dtype.is_floating_point:
        raise TypeError(f"Expected a floating point output, but got {output.dtype}")

    # make sure output is finite
    if not torch.all(torch.isfinite(output)):
        raise ValueError("Output has NAN and or INF values!")

    return output


@torch.no_grad()
def is_forward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    num_runs: int = 100,
    tol: Optional[float] = None,
) -> bool:
    r"""Check if the forward pass is stable.

    Assumptions:

    - The module supports batching.
    - The module takes a fixed size nu
    - The module returns a single tensor.

    The test works as follows:

    1. Compute the means μ and standard deviations σ of the output for a large number of random inputs.
    2. For each output, consider the distance between the input distribution $𝓝(0, 1)$,
       and output distribution $𝓝(μ, σ²)$. We measure this distance in terms of some divergence measure
       such as KL-divergence, Wasserstein distance, etc.
    3. We test relative closeness via the formula:

    .. math:: \dist(𝓝(0, 1), 𝓝(μ, σ²)) ≤ rtol⋅mag(𝓝(0, 1)) + atol

    where dist is some divergence measure and mag is measure of the magnitude of the distribution.

    More specifically, we consider the entropy:

     .. math:: H(p,q) - H(p) = d(p, q) ≤ rtol⋅H(q) + atol

    In the special case when $p=𝓝(μ,σ²)$ and $q=𝓝(0,1)$ are univariate gaussian, we have:

    .. math:: \dist(𝓝(μ,σ²), 𝓝(0,1)) ≤ rtol⋅H(𝓝(0,1)) + atol \\
        ⟺ ½(μ² + σ² - 1 - \log(σ²)) ≤ rtol⋅½(1 + \log(2π)) + atol

    Recall the following facts about the information content of normal distributions:

    1. (univariate entropy) $H(𝓝(μ, σ²)) = ½\log(2πeσ²)$
    2. (univariate KL) $\KL(p₁, p₂) = ½(σ₁²/σ₂² + (μ₁ - μ₂)²/σ₂² + \log(σ₂²/σ₁²) - 1)$
        - if $σ₁² = σ₂²$, then $\KL(p₁, p₂) = ½(μ₁ - μ₂)²$
            - > Test A: $\dist(p,q) < ε$ is satisfied if and only if $\abs{μ₁ - μ₂} < ε$
            - > Test B: $\dist(p,q) < β⋅H(q)+α$ is satisfied if and only if $\abs{μ₁ - μ₂} < β̃\log(σ) + α$
                If $σ → 0$, then the test becomes more difficult, and even potentially impossible.
                If $σ → ∞$, then the test becomes easier.
        - if $σ₂ ≫ 1$, then $\KL(p₁, p₂) ≈ 𝓞(\log(σ₂))$
    3. univariate Wasserstein distance: $W₂(p₁, p₂)² = \abs{μ₁ - μ₂}² + \abs{σ₁ - σ₂}²$

    In particular, consider the case when we have two zero-centered normal distributions $𝓝(0, σ₁²)$ and $𝓝(0, σ₂²)$.
    If we increase the standard deviation of the reference distribution,
    then the KL-divergence increases as $𝓞(\log(σ₂))$, but also the entropy increases as $𝓞(\log(σ₂))$.

    Is this true generally? (I.e. does this make the test "entropy-stable"?)
    """
    # generate random N(0,1) inputs
    inputs = [torch.randn(num_runs, *shape) for shape in input_shapes]
    output = get_output(func, *inputs)
    dims = list(range(1, output.ndim))
    result = is_standardized(output, dim=dims, tol=tol)
    return bool(result.all().item())


@torch.no_grad()
def is_backward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    check_params: bool = False,
    num_runs: int = 100,
    tol: Optional[float] = None,
) -> bool:
    r"""Check if a function is backward stable.

    In this context, a function is called backward stable, if its vector jacobian product,
    i.e. the function $v↦vᵀ(∂f/∂x)$ is forward stable (at a given point $x$).

    To test backward stability, we randomly sample $x∼𝓝(0,1)$ and $v∼𝓝(0,1)$
    with the same shape as $f(x)$. Then we call `.backward()` on the scalar value `⟨v, f(x)⟩`.
    We then check whether `x.grad` has zero mean and unit variance.
    """
    # generate random N(0,1) inputs
    inputs = [
        torch.randn(num_runs, *shape, requires_grad=True) for shape in input_shapes
    ]

    with torch.enable_grad():
        output = get_output(func, *inputs)
        v = torch.randn_like(output)
        loss = (v * output).sum()
        loss.backward()

    passed = True

    # check input gradients
    assert all(x.grad is not None for x in inputs)
    input_grads = [x.grad for x in inputs if x.grad is not None]

    passed &= all(
        is_standardized(g, dim=g.shape[1:], tol=tol).all().item() for g in input_grads
    )

    # check parameter gradients
    if check_params:
        if not isinstance(func, nn.Module):
            raise TypeError(f"Expected a module, got {type(func)}")
        param_grads = (p.grad for p in func.parameters() if p.grad is not None)
        passed &= all(
            is_standardized(g, dim=g.shape, tol=tol).item() for g in param_grads
        )

    return passed


@torch.no_grad()
def assert_forward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    num_runs: int = 100,
    tol: Optional[float] = None,
) -> None:
    r"""Raises AssertionError if the forward pass is not stable."""
    if not is_forward_stable(func, input_shapes, num_runs=num_runs, tol=tol):
        raise AssertionError(
            f"Function is not forward stable (tolerance: {tol}, runs: {num_runs})"
        )


@torch.no_grad()
def assert_backward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    num_runs: int = 100,
    check_params: bool = False,
    tol: Optional[float] = None,
) -> None:
    r"""Raises AssertionError if the function is not backward stable."""
    if not is_backward_stable(
        func,
        input_shapes,
        num_runs=num_runs,
        check_params=check_params,
        tol=tol,
    ):
        raise AssertionError(
            f"Function is not backward stable (tolerance: {tol}, runs: {num_runs})"
        )
