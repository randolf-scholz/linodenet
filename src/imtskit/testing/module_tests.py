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
    "is_standardized",
]

from collections.abc import Callable, Sequence
from typing import Optional, Protocol, SupportsFloat

import torch
from torch import Tensor, nn

from imtskit.constants import ATOL, RTOL
from signatures import signature


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


def _get_dims(dim: None | int | Sequence[int], values: Tensor) -> list[int]:
    return (
        [dim]
        if isinstance(dim, int)
        else list(range(values.ndim))
        if dim is None
        else list(dim)
    )


def _get_tol(tol: float | None, values: Tensor, *, dims: list[int]) -> float:
    if isinstance(tol, SupportsFloat):
        return float(tol)

    # default: 3-sigma rule
    output_lengths = torch.tensor([values.shape[k] for k in dims])
    count = output_lengths.prod()
    tol = 3.0 / count.sqrt().item()
    return tol


@signature("(..., *ds) -> (...)")
def is_standardized(
    values: Tensor,  # Float[..., *ds]
    /,
    *,
    dim: None | int | tuple[int, ...] | list[int] = -1,
    tol: Optional[float] = None,
) -> Tensor:  # Float[...]
    r"""Check if a tensor has zero mean and unit variance.

    Args:
        values: The tensor to check.
        dim: the axis over which to compute the mean and stdv.
        tol: the tolerance

    Note:
        Often, normality will be achieved approximately, in terms of the CTL.
        As the sample mean of $n$-many samples from a normal distribution is
        distributed as $N(μ, σ²/n)$, knowing that the input should be $N(0, 1)$,
        we can expect the sample mean to be distributed as $N(0, 1/n)$,
        that is with standard deviation $1/√n$.
        Therefore, to get $k$-sigma confidence, we should check whether the mean is
        inside the interval $[-k/√n, k/√n]$.
    """
    dims = _get_dims(dim, values)

    # compute mean an stdv
    tol = _get_tol(tol, values, dims=dims)

    mean_values = values.mean(dim=dims)
    stdv_values = values.std(dim=dims)

    # check that the mean is close to 0 and stdv is close to 1
    mean_valid = mean_values.abs() <= tol
    stdv_valid = (stdv_values - 1.0).abs() <= tol
    return mean_valid & stdv_valid


@torch.no_grad()
def is_forward_stable(
    func: Callable[..., Tensor],
    input_shapes: list[tuple[int, ...]],
    *,
    num_runs: int = 100,
    tol: Optional[float] = None,
) -> bool:
    r"""Check if the function is forward stable.

    By definition, this is the case if, when given random zero mean and unit variance data,
    the function returns values with zero mean and unit variance results.

    Given $f：ℝⁿ→ℝᵐ$, sample $xᵢ⁽ᵏ⁾∼𝓝(0,1)$, and compute $y⁽ᵏ⁾=f(x⁽ᵏ⁾)$.
    Then
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
    with the same shape as $f(x)$. Then we call ``.backward()`` on the scalar value $⟨v, f(x)⟩$.
    We then check whether ``x.grad`` has zero mean and unit variance.
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
    r"""Raises AssertionError if the function is not forward stable."""
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
