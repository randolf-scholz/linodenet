r"""Checks for testing certain module properties."""

__all__ = [
    # ABCs & Protocols
    "ModuleTest",
    # Functions
    "is_forward_stable",
    "is_backward_stable",
    "check_zero_mean_unit_variance",
    "get_output",
]

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor, nn

from linodenet.constants import ATOL, ONE, RTOL, ZERO


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
        raise RuntimeError(f"Error in forward pass of {func}") from exc

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
def check_zero_mean_unit_variance(
    values: Tensor,
    /,
    *,
    batch_dim: int = 0,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> bool:
    r"""Check if a tensor has zero mean and unit variance."""
    # compute mean an stdv
    output_dims = tuple(k for k in range(values.ndim) if k != batch_dim)
    mean_values = values.mean(dim=output_dims)
    stdv_values = values.std(dim=output_dims)

    # check if mean is close to 0 and stdv is close to 1 **across all runs** using RMSE
    # we use RMSE instead of just regular norm since we want to test if the value is close **on average**
    # since the convergence rate via the law of large numbers is slow (often only O(1/√n)),
    # It seems fine to use a test where sample size is discounted.
    mean_rmse = (mean_values - ZERO).abs().pow(2).mean().sqrt()
    stdv_rmse = (stdv_values - ONE).abs().pow(2).mean().sqrt()
    mean_valid = mean_rmse <= (rtol * ZERO + atol)
    stdv_valid = stdv_rmse <= (rtol * ONE + atol)

    return bool(mean_valid) and bool(stdv_valid)


@torch.no_grad()
def is_forward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    num_runs: int = 100,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    r"""Check if the forward pass is stable.

    Assumptions:
      - The module supports batching.
      - The module takes a fixed size nu
      - The module returns a single tensor.

    The test works as follows:

    1. Compute the means μ and standard deviations σ of the output for a large number of random inputs.
    2. For each output, consider the distance between the input distribution 𝓝(0, 1) and output distribution 𝓝(μ, σ²).
       We measure this distance in terms of some divergence measure (e.g. KL-divergence, Wasserstein distance, etc.).
    3. We test relative closeness via the formula:

    .. math:: dist(N(0, 1), N(μ, σ²)) ≤ rtol⋅mag(N(0, 1)) + atol

    where dist is some divergence measure and mag is measure of the magnitude of the distribution.

    More specifically, we consider the entropy:

     .. math:: H(p,q) - H(p) = d(p, q) ≤ rtol⋅H(q) + atol

    In the special case when p=N(μ, σ²) and q=N(0, 1) are univariate gaussian, we have:

    .. math:: d(N(μ, σ²), N(0, 1)) ≤ rtol⋅H(N(0, 1)) + atol \\
        ⟺ ½(μ² + σ² - 1 - \log(σ²)) ≤ rtol⋅½(1 + \log(2π)) + atol



    Recall the following facts about the information content of normal distributions:

    1. (univariate entropy) H(N(μ, σ²)) = ½\log(2πeσ²)
    2. (univariate KL) KL(p₁, p₂) = ½(σ₁²/σ₂² + (μ₁ - μ₂)²/σ₂² + \log(σ₂²/σ₁²) - 1)
       - if σ₁² = σ₂², then KL(p₁, p₂) = ½(μ₁ - μ₂)²
            -> Test A: d(p,q)<ε is satisfied if and only if |μ₁ - μ₂| < ε
            -> Test B: d(p,q)<β H(q) + α is satisfied if and only if |μ₁ - μ₂| < β̃\log(σ) + α
               If σ → 0, then the test becomes more difficult, and even potentially impossible.
               If σ → ∞, then the test becomes easier.
       - if σ₂>>1, then KL(p₁, p₂) ≈ O(\log(σ₂))
    3. univariate Wasserstein distance: W₂(p₁, p₂)² = |μ₁ - μ₂|² + |σ₁ - σ₂|²

    In particular, consider the case when we have two zero-centered normal distributions N(0, σ₁²) and N(0, σ₂²).
    If we increase the standard deviation of the reference distribution,
    then the KL-divergence increases as $O(\log(σ₂))$, but also the entropy increases as $O(\log(σ₂))$.

    Is this true generally? (I.e. does this make the test "entropy-stable"?)
    """
    # generate random N(0,1) inputs
    inputs = [torch.randn(num_runs, *shape) for shape in input_shapes]
    output = get_output(func, *inputs)
    return check_zero_mean_unit_variance(output, rtol=rtol, atol=atol)


@torch.no_grad()
def is_backward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    check_params: bool = False,
    num_runs: int = 100,
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
    input_grads = (x.grad for x in inputs if x.grad is not None)
    passed &= all(
        check_zero_mean_unit_variance(grad, rtol=rtol, atol=atol)
        for grad in input_grads
    )

    if check_params:
        if not isinstance(func, nn.Module):
            raise TypeError(f"Expected a module, got {type(func)}")
        param_grads = (p.grad for p in func.parameters() if p.grad is not None)
        passed &= all(
            check_zero_mean_unit_variance(grad, rtol=rtol, atol=atol)
            for grad in param_grads
        )

    return passed


@torch.no_grad()
def assert_forward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    num_runs: int = 100,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    r"""Check if the forward pass is stable.

    Raises:
        AssertionError: If the forward pass is not stable.
    """
    # generate random N(0,1) inputs
    inputs = [torch.randn(num_runs, *shape) for shape in input_shapes]
    output = get_output(func, *inputs)
    assert check_zero_mean_unit_variance(output, rtol=rtol, atol=atol)


@torch.no_grad()
def assert_backward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    check_params: bool = False,
    num_runs: int = 100,
) -> None:
    r"""Check if a function is backward stable.

    Raises:
        AssertionError: If the function is not backward stable.
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

    # check input gradients
    assert all(x.grad is not None for x in inputs)
    input_grads = (x.grad for x in inputs if x.grad is not None)

    for grad in input_grads:
        assert check_zero_mean_unit_variance(grad, rtol=rtol, atol=atol)

    if check_params:
        if not isinstance(func, nn.Module):
            raise TypeError(f"Expected a module, got {type(func)}")
        param_grads = (p.grad for p in func.parameters() if p.grad is not None)
        for grad in param_grads:
            assert check_zero_mean_unit_variance(grad, rtol=rtol, atol=atol)
