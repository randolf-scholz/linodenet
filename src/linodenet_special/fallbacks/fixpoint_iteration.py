r"""Fixed point iteration with implicit differentiation."""

__all__ = [
    "FixpointSolve",
    "FixpointState",
    "fallback_iteration",
    "fixpoint_condition",
    "fixpoint_iteration",
    "fixpoint_solve",
    "fixpoint_solve_functional",
]

import warnings
from collections.abc import Callable
from typing import Any, Concatenate, Final, NamedTuple

import torch
from torch import Tensor, nn

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

        # SEC: return ∂y/∂x = (∂f/∂θ)ᵀu⁎
        _, params_vjp_fn = torch.func.vjp(lambda *θ: ctx.fn(x_star, *θ), *params)  # pyright: ignore[reportAssignmentType]
        grad_params = params_vjp_fn(u_star)
        grad_initial_guess = torch.zeros_like(x_star)

        return None, grad_initial_guess, None, None, None, *grad_params


def fixpoint_solve_functional(
    fn: Callable[..., Tensor],
    x0: Tensor,
    /,
    *args: Tensor,
    maxiter: int = _DEFAULT_MAXITER,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tensor:
    r"""Solve $x = f(x, *ys)$ by fixed point iteration.

    Args:
        fn: Mapping defining the fixed point equation $x = f(x, θ)$.
            The callable must accept `x` as its first argument and any tensor
            parameters passed through `*params` afterwards.
        x0: Starting point of the iteration.
        *args: Tensor parameters passed through to `fn`.
        maxiter: Maximum number of fixed point iterations used in both forward
            and backward solves.
        atol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
    """
    return FixpointSolve.apply(fn, x0, maxiter, atol, rtol, *args)  # pyright: ignore[reportReturnType]


def _warn_if_not_converged(
    budget: Tensor,
    residual: Tensor,
    maxiter: Tensor,
    atol: Tensor,
    /,
) -> None:
    if budget <= 0:
        warnings.warn(
            f"No convergence in {maxiter} iterations."
            f"Final residual: {residual.max()} > {atol}.",
            stacklevel=3,
        )


def _fixpoint_solve_impl(
    fn: Callable[Concatenate[Tensor, ...], Tensor],
    x0: Tensor,
    /,
    *params,
    maxiter: int,
    atol: float,
    rtol: float,
) -> Tensor:
    maxiter_tensor = torch.as_tensor(maxiter, dtype=torch.int32, device=x0.device)
    atol_tensor = torch.as_tensor(atol, dtype=x0.dtype, device=x0.device)
    rtol_tensor = torch.as_tensor(rtol, dtype=x0.dtype, device=x0.device)
    r0 = torch.full_like(x0, torch.inf)
    initial_state = FixpointState(maxiter_tensor, x0, r0, atol_tensor, rtol_tensor)
    budget, x_star, residual, _, _ = fixpoint_iteration(
        lambda z: fn(z, *params), initial_state
    )
    _warn_if_not_converged(budget, residual, maxiter_tensor, atol_tensor)
    return x_star


def _fallback_solve_impl(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    /,
    *,
    maxiter: int,
    atol: float,
    rtol: float,
) -> Tensor:
    maxiter_tensor = torch.as_tensor(maxiter, dtype=torch.int32, device=x0.device)
    atol_tensor = torch.as_tensor(atol, dtype=x0.dtype, device=x0.device)
    rtol_tensor = torch.as_tensor(rtol, dtype=x0.dtype, device=x0.device)
    r0 = torch.full_like(x0, torch.inf)
    initial_state = FixpointState(maxiter_tensor, x0, r0, atol_tensor, rtol_tensor)
    budget, x_star, residual, _, _ = fallback_iteration(fn, initial_state)
    _warn_if_not_converged(budget, residual, maxiter_tensor, atol_tensor)
    return x_star


def fixpoint_solve(
    fn: Callable[Concatenate[Tensor, ...], Tensor] | nn.Module,
    x0: Tensor,
    /,
    *args: Tensor,
    maxiter: int = _DEFAULT_MAXITER,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tensor:
    r"""Solve $x = f(x, *ys)$ by fixed point iteration."""
    with torch.no_grad():
        x_star = _fixpoint_solve_impl(
            fn, x0, *args, maxiter=maxiter, atol=atol, rtol=rtol
        )

    if not torch.is_grad_enabled():
        return x_star

    # re-engage gradients coming from fn itself at the fixed point.
    x_star = fn(x_star, *args)
    if not x_star.requires_grad:
        return x_star

    # set up the backward hook.

    x_ref = x_star.detach().clone().requires_grad_(True)
    f_ref = fn(x_ref, *args)

    @torch.no_grad()
    def backward_hook(grad: Tensor | None, /) -> Tensor | None:
        if grad is None:
            return None

        # SEC: solve u = g + (∂f/∂x)ᵀu by fixed point iteration
        # SEC: return ∂y/∂x = (∂f/∂θ)ᵀu⁎
        return _fallback_solve_impl(
            lambda u: grad + torch.autograd.grad(f_ref, x_ref, u, retain_graph=True)[0],
            grad,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
        )

    x_star.register_hook(backward_hook)
    return x_star
