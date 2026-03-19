r"""Fixed point iteration with implicit differentiation."""

__all__ = [
    "FixpointSolve",
    "FixpointState",
    "fallback_iteration",
    "fixpoint_condition",
    "fixpoint_iteration",
    "fixpoint_solve",
]

import warnings
from collections.abc import Callable
from typing import Any, Final, NamedTuple

import torch
from torch import Tensor

_DEFAULT_MAXITER: Final[int] = 256


class FixpointState(NamedTuple):
    r"""Loop state for forward fixed-point iteration."""

    budget: Tensor
    x: Tensor
    residual: Tensor
    atol: Tensor
    rtol: Tensor


def fixpoint_condition(state: FixpointState, /) -> Tensor:
    budget, x, residual, atol, rtol = state
    tolerance = rtol * x.abs() + atol
    return (budget > 0) & (residual > tolerance).any()


def _python_while_loop(
    cond_fn: Callable[[FixpointState], Tensor | bool],
    body_fn: Callable[[FixpointState], FixpointState],
    state: FixpointState,
    /,
) -> FixpointState:
    while cond_fn(state):
        state = body_fn(state)
    return state


@torch.no_grad()
def fixpoint_iteration(
    fn: Callable[[Tensor], Tensor], initial_state: FixpointState, /
) -> FixpointState:
    r"""Solve $x = f(x, θ)$ by fixed point iteration."""

    def body_fn(state: FixpointState, /) -> FixpointState:
        budget, x_prev, _, atol, rtol = state
        x = fn(x_prev)
        r = (x - x_prev).abs()
        return FixpointState(budget - 1, x, r, atol.clone(), rtol.clone())

    return torch.while_loop(fixpoint_condition, body_fn, (initial_state,))  # pyright: ignore[reportReturnType]


@torch.no_grad()
def fallback_iteration(
    fn: Callable[..., Tensor], initial_state: FixpointState, /
) -> FixpointState:
    r"""Fixed point iteration with plain python loop."""

    def body_fn(state: FixpointState, /) -> FixpointState:
        budget, x_prev, _, atol, rtol = state
        x = fn(x_prev)
        r = (x - x_prev).abs()
        return FixpointState(budget - 1, x, r, atol.clone(), rtol.clone())

    # FIXME: torch.while_loop with VJP raises UncapturedHigherOrderOpError
    return _python_while_loop(fixpoint_condition, body_fn, initial_state)


class FixpointSolve(torch.autograd.Function):
    r"""Solve a fixed point equation with implicit differentiation.

    The forward pass solves $x = f(x, θ)$ for the fixed point $x$ by iteration.

    The backward pass uses the implicit function theorem. For an upstream
    cotangent $g$, it solves

    .. math:: u = g + (∂f/∂x)ᵀu

    and then returns the parameter gradients $(∂f/∂θ)ᵀu$.

    Notes:
        This implementation is intended for eager-mode experimentation.
        It does not differentiate with respect to the initial guess.
    """

    @staticmethod
    def forward(
        ctx: Any,
        fn: Callable[..., Tensor],
        x0: Tensor,
        maxiter: Tensor,
        atol: Tensor,
        rtol: Tensor,
        /,
        *params: Tensor,
    ) -> Tensor:
        ctx.fn = fn
        ctx.maxiter = torch.as_tensor(maxiter, dtype=torch.int32, device=x0.device)
        ctx.atol = torch.as_tensor(atol, dtype=x0.dtype, device=x0.device)
        ctx.rtol = torch.as_tensor(rtol, dtype=x0.dtype, device=x0.device)

        # SEC: solve x = f(x, θ) with fixed point iteration
        r0 = torch.full_like(x0, torch.inf)
        initial_state = FixpointState(ctx.maxiter, x0, r0, ctx.atol, ctx.rtol)
        sol = fixpoint_iteration(lambda z: fn(z, *params), initial_state)
        budget, x_star, residual, _, _ = sol

        if budget <= 0:
            warnings.warn(
                f"No convergence in {ctx.maxiter} iterations."
                f"Final residual: {residual.max()} > {ctx.atol}.",
                stacklevel=2,
            )

        ctx.save_for_backward(x_star, *params)
        return x_star

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[None, Tensor, None, None, None, *tuple[Tensor | None, ...]]:
        (grad_output,) = grad_outputs
        x_star, *params = ctx.saved_tensors

        if grad_output is None:
            grad_output = torch.zeros_like(x_star)

        # SEC: solve u = g + (∂f/∂x)ᵀu by fixed point iteration
        _, vjp_fn = torch.func.vjp(lambda x: ctx.fn(x, *params), x_star)  # pyright: ignore[reportAssignmentType]
        r0 = torch.full_like(x_star, torch.inf)
        initial_state = FixpointState(ctx.maxiter, grad_output, r0, ctx.atol, ctx.rtol)
        sol = fallback_iteration(lambda u: grad_output + vjp_fn(u)[0], initial_state)
        budget, u_star, residual, _, _ = sol

        if budget <= 0:
            warnings.warn(
                f"No convergence in {ctx.maxiter} iterations."
                f"Final residual: {residual.max()} > {ctx.atol}.",
                stacklevel=2,
            )

        # ∂y/∂x = (∂f/∂θ)ᵀu⁎
        _, params_vjp_fn = torch.func.vjp(lambda *θ: ctx.fn(x_star, *θ), *params)  # pyright: ignore[reportAssignmentType]
        grad_params = params_vjp_fn(u_star)
        grad_initial_guess = torch.zeros_like(x_star)

        return None, grad_initial_guess, None, None, None, *grad_params


def fixpoint_solve(
    fn: Callable[..., Tensor],
    x0: Tensor,
    /,
    *params: Tensor,
    maxiter: int = _DEFAULT_MAXITER,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tensor:
    r"""Solve $x = f(x, θ)$ by fixed point iteration.

    Args:
        fn: Mapping defining the fixed point equation $x = f(x, θ)$.
            The callable must accept `x` as its first argument and any tensor
            parameters passed through `*params` afterwards.
        x0: Starting point of the iteration.
        *params: Tensor parameters passed through to `fn`.
        maxiter: Maximum number of fixed point iterations used in both forward
            and backward solves.
        atol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
    """
    return FixpointSolve.apply(fn, x0, maxiter, atol, rtol, *params)  # pyright: ignore[reportReturnType]
