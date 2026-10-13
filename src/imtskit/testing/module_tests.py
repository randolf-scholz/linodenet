r"""Checks for tensor standardization and module forward and backward stability."""

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
    r"""Protocol for callable module checks with relative and absolute tolerances."""

    def __call__(
        self,
        module: nn.Module,
        /,
        *,
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> bool:
        r"""Return whether the module passes the check at the given tolerances."""
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
    r"""Check whether a tensor has approximately zero mean and unit variance.

    Args:
        values: The tensor to check.
        dim: The axis or axes over which to compute the mean and standard
            deviation. `None` reduces over all axes.
        tol: The absolute tolerance $τ$ for both statistics. Defaults to $3/√n$,
            where $n$ is the number of elements along the reduced axes.

    Returns:
        A boolean tensor over the remaining axes, indicating whether both
        $|μ| ≤ τ$ and $|s - 1| ≤ τ$ hold, where $s$ is the sample standard
        deviation computed with Bessel's correction.

    Note:
        For $n$ independent samples from $𝓝(0, 1)$, the sample mean has
        distribution $𝓝(0, 1/n)$ and standard deviation $1/√n$. The central
        limit theorem (CLT) gives an approximate version for other distributions
        with zero mean and unit variance. The default tolerance uses this
        three-sigma heuristic for the mean; applying it to the standard deviation does not
        give the same confidence level. This check does not test normality.

        Slices with fewer than two elements have an undefined sample standard
        deviation and fail the check. NaN statistics also fail.
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
    r"""Check whether a function preserves approximate standardization.

    In this context, forward stability means that standard normal inputs produce
    outputs with approximately zero mean and unit variance within each batch item.

    Given $f: ℝⁿ → ℝᵐ$, sample $xᵢ⁽ᵏ⁾ ∼ 𝓝(0, 1)$ independently and compute
    $y⁽ᵏ⁾ = f(x⁽ᵏ⁾)$. Each output $y⁽ᵏ⁾$ must pass `is_standardized` over
    all of its non-batch axes.

    Args:
        func: The function or module to check. It must return a finite floating
            point tensor with at least two dimensions and preserve the batch size.
        input_shapes: The input tensor shapes, excluding the leading batch axis.
            At least one input is required.
        num_runs: The number of independent samples in the batch.
        tol: The tolerance passed to `is_standardized`.

    Returns:
        Whether every sampled output passes the standardization check.
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
    r"""Check whether a function's gradients are approximately standardized.

    In this context, backward stability means that the vector-Jacobian product
    $v ↦ vᵀ(∂f/∂x)$ preserves approximate standardization at sampled points $x$.

    Sample the entries of $x$ and $v$ independently from $𝓝(0, 1)$, with $v$
    shaped like $f(x)$. Call `.backward()` on the scalar $⟨v, f(x)⟩$, then check
    whether `x.grad` has approximately zero mean and unit variance.

    Args:
        func: The function or module to check. It must return a finite floating
            point tensor with at least two dimensions and preserve the batch size.
        input_shapes: The input tensor shapes, excluding the leading batch axis.
            At least one input is required, and every input must receive a gradient.
        check_params: Whether to also check parameter gradients. Requires `func`
            to be a module; parameters without gradients are skipped.
        num_runs: The number of independent samples in the batch.
        tol: The tolerance passed to `is_standardized`.

    Returns:
        Whether all checked gradients pass the standardization check.

    Note:
        The backward pass accumulates gradients in any module parameters.
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
    r"""Raise `AssertionError` if the function is not forward stable."""
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
    r"""Raise `AssertionError` if the function is not backward stable."""
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
