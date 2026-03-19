r"""Fixed point iteration with implicit differentiation."""

__all__ = ["FixpointSolve", "fixpoint_solve"]

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


def fixpoint_cond(state: FixpointState, /) -> Tensor:
    budget, x, residual, atol, rtol = state
    tolerance = rtol * x.abs() + atol
    return (budget > 0) & (residual > tolerance).any()


@torch.no_grad()
def _fixed_point_iteration(
    fn: Callable[..., Tensor],
    initial_guess: Tensor,
    /,
    *params: Tensor,
    maxiter: Tensor,
    atol: Tensor,
    rtol: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Solve $x = f(x, θ)$ by fixed point iteration."""

    def body_fn(state: FixpointState, /) -> FixpointState:
        budget, x_prev, _, atol, rtol = state
        x = fn(x_prev, *params)
        residual = torch.abs(x - x_prev)
        return FixpointState(budget - 1, x, residual, atol.clone(), rtol.clone())

    r0 = torch.full_like(initial_guess, torch.inf)
    initial_state = FixpointState(maxiter, initial_guess, r0, atol, rtol)
    final_state = torch.while_loop(fixpoint_cond, body_fn, (initial_state,))
    budget, x, _, _, _ = final_state  # pyright: ignore[reportGeneralTypeIssues]
    converged = budget > 0
    return x, converged


@torch.no_grad()
def _implicit_vjp(
    vjp_fn: Callable[..., Tensor],
    grad_output: Tensor,
    /,
    maxiter: Tensor,
    atol: Tensor,
    rtol: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Solve $u = g + (∂f/∂x)ᵀu$ using the `while cond(state): state = body(state)` schema."""

    def body_fn(state: FixpointState, /) -> FixpointState:
        budget, u, _, atol, rtol = state
        (vjp_x,) = vjp_fn(u)
        u_next = grad_output + vjp_x
        residual = torch.abs(u_next - u)
        return FixpointState(budget - 1, u_next, residual, atol.clone(), rtol.clone())

    # FIXME: torch.while_loop with VJP raises UncapturedHigherOrderOpError
    r0 = torch.full_like(grad_output, torch.inf)
    state = FixpointState(maxiter, grad_output, r0, atol, rtol)
    while fixpoint_cond(state):
        state = body_fn(state)

    budget, u, _, _, _ = state
    converged = budget > 0

    return u, converged


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
        initial_guess: Tensor,
        maxiter: Tensor,
        atol: Tensor,
        rtol: Tensor,
        /,
        *params: Tensor,
    ) -> Tensor:
        ctx.fn = fn
        ctx.maxiter = maxiter
        ctx.atol = atol
        ctx.rtol = rtol

        x_star, converged = _fixed_point_iteration(
            fn,
            initial_guess,
            *params,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
        )
        if not converged:
            warnings.warn(
                f"No convergence in {int(torch.as_tensor(maxiter).item())} iterations.",
                stacklevel=2,
            )

        ctx.save_for_backward(initial_guess, x_star, *params)
        return x_star

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[None, Tensor, None, None, None, *tuple[Tensor | None, ...]]:
        (grad_output,) = grad_outputs
        initial_guess, x_star, *params = ctx.saved_tensors

        if grad_output is None:
            grad_output = torch.zeros_like(x_star)

        with torch.enable_grad():
            x = x_star.detach().requires_grad_(True)
            differentiable_params = tuple(
                param.detach().requires_grad_(True) for param in params
            )
            fx, vjp_fn = torch.func.vjp(
                lambda z: ctx.fn(z, *differentiable_params),
                x,
            )

            u, converged = _implicit_vjp(
                vjp_fn,
                grad_output,
                maxiter=ctx.maxiter,
                atol=ctx.atol,
                rtol=ctx.rtol,
            )
            grad_params = torch.autograd.grad(
                fx,
                differentiable_params,
                grad_outputs=u,
                allow_unused=True,
            )

        if not converged:
            warnings.warn(
                f"No backward convergence in {ctx.maxiter} iterations.",
                stacklevel=2,
            )

        grad_initial_guess = torch.zeros_like(initial_guess)
        return None, grad_initial_guess, None, None, None, *grad_params


def fixpoint_solve(
    fn: Callable[..., Tensor],
    initial_guess: Tensor,
    /,
    *params: Tensor,
    maxiter: int | Tensor = _DEFAULT_MAXITER,
    atol: float | Tensor = 1e-6,
    rtol: float | Tensor = 1e-6,
) -> Tensor:
    r"""Solve $x = f(x, θ)$ by fixed point iteration.

    Args:
        fn: Mapping defining the fixed point equation $x = f(x, θ)$.
            The callable must accept `x` as its first argument and any tensor
            parameters passed through `*params` afterwards.
        initial_guess: Starting point of the iteration.
        *params: Tensor parameters passed through to `fn`.
        maxiter: Maximum number of fixed point iterations used in both forward
            and backward solves.
        atol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
    """
    atol = torch.as_tensor(atol, dtype=initial_guess.dtype, device=initial_guess.device)
    rtol = torch.as_tensor(rtol, dtype=initial_guess.dtype, device=initial_guess.device)
    maxiter = torch.as_tensor(maxiter, dtype=torch.int32, device=initial_guess.device)
    return FixpointSolve.apply(fn, initial_guess, maxiter, atol, rtol, *params)  # pyright: ignore[reportReturnType]
