r"""Fixed point iteration with implicit differentiation."""

__all__ = [
    "FixpointSolution",
    "fixpoint_solve",
    "fixpoint_solve_functional",
]

from collections.abc import Callable
from typing import Any, Concatenate, Final, NamedTuple

import torch
from torch import Tensor, nn

_DEFAULT_MAXITER: Final[int] = 256


class FixpointSolution(NamedTuple):
    r"""Fixed point solution."""

    x: Tensor
    residual: Tensor
    budget: int

    maxiter: int
    atol: float
    rtol: float


class _LoopState(NamedTuple):
    r"""Loop state for forward fixed-point iteration."""

    x: Tensor
    residual: Tensor
    budget: Tensor


def _python_while_loop(
    cond_fn: Callable[[_LoopState], Tensor | bool],
    body_fn: Callable[[_LoopState], _LoopState],
    state: _LoopState,
    /,
) -> _LoopState:
    while cond_fn(state):
        state = body_fn(state)
    return state


@torch.no_grad()
def _fixpoint_solve_impl(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    /,
    maxiter: Tensor,
    atol: Tensor,
    rtol: Tensor,
) -> _LoopState:
    def cond_fn(state: _LoopState, /) -> Tensor:
        x, residual, budget = state
        tolerance = rtol * x.abs() + atol
        return (budget > 0) & (residual > tolerance).any()

    def body_fn(state: _LoopState, /) -> _LoopState:
        x_prev, _, budget = state
        x = fn(x_prev)
        r = (x - x_prev).abs()
        return _LoopState(x, r, budget - 1)

    r0 = torch.full_like(x0, torch.inf)
    initial_state = _LoopState(x0, r0, maxiter)

    return torch.while_loop(cond_fn, body_fn, (initial_state,))


@torch.no_grad()
def _fallback_solve_impl(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    /,
    *,
    maxiter: Tensor,
    atol: Tensor,
    rtol: Tensor,
) -> _LoopState:
    def cond_fn(state: _LoopState, /) -> Tensor:
        x, residual, budget = state
        tolerance = rtol * x.abs() + atol
        return (budget > 0) & (residual > tolerance).any()

    def body_fn(state: _LoopState, /) -> _LoopState:
        x_prev, _, budget = state
        x = fn(x_prev)
        r = (x - x_prev).abs()
        return _LoopState(x, r, budget - 1)

    r0 = torch.full_like(x0, torch.inf)
    initial_state = _LoopState(x0, r0, maxiter)
    final_state: _LoopState = _python_while_loop(
        cond_fn,
        body_fn,
        initial_state,
    )
    return final_state


class _FixpointSolve_Impl(torch.autograd.Function):
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
        fn: Callable[..., Tensor],
        x0: Tensor,
        maxiter: Tensor,
        atol: Tensor,
        rtol: Tensor,
        *params: Tensor,
    ) -> Tensor:
        # SEC: solve x = f(x, θ) with fixed point iteration
        sol = _fixpoint_solve_impl(
            lambda z: fn(z, *params),
            x0,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
        )
        return sol.x

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        fn, _, maxiter, atol, rtol, *params = inputs
        ctx.save_for_backward(output, maxiter, atol, rtol, *params)
        ctx.fn = fn

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[None, Tensor, None, None, None, *tuple[Tensor | None, ...]]:
        (grad_output,) = grad_outputs
        x_star, maxiter, atol, rtol, *params = ctx.saved_tensors

        if grad_output is None:
            grad_output = torch.zeros_like(x_star)

        # FIXME: https://github.com/pytorch/pytorch/issues/179510
        #  cant use torch.func.vjp due to torch.compile bug.
        with torch.enable_grad():
            z = ctx.fn(x_star, *params)
            z0 = z.clone().detach().requires_grad_(True)
            f0 = ctx.fn(z0, *params)
            vjp_fn = lambda u: torch.autograd.grad(f0, z0, u, retain_graph=True)  # noqa: E731

        # SEC: solve u = g + (∂f/∂x)ᵀu by fixed point iteration
        sol = _fallback_solve_impl(
            lambda u: grad_output + vjp_fn(u)[0],
            grad_output,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
        )

        # SEC: return ∂y/∂x = (∂f/∂θ)ᵀu⁎
        _, params_vjp_fn, *_ = torch.func.vjp(lambda *θ: ctx.fn(x_star, *θ), *params)
        grad_params = params_vjp_fn(sol.x)

        return None, torch.zeros_like(x_star), None, None, None, *grad_params


def fixpoint_solve_functional(
    fn: Callable[..., Tensor],
    x0: Tensor,
    *,
    args: tuple[Tensor, ...] = (),
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
        args: Tensor parameters passed through to `fn`.
        maxiter: Maximum number of fixed point iterations used in both forward
            and backward solves.
        atol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
    """
    maxiter_t = torch.as_tensor(maxiter, dtype=torch.int32, device=x0.device)
    atol_t = torch.as_tensor(atol, dtype=x0.dtype, device=x0.device)
    rtol_t = torch.as_tensor(rtol, dtype=x0.dtype, device=x0.device)
    return _FixpointSolve_Impl.apply(fn, x0, maxiter_t, atol_t, rtol_t, *args)


def fixpoint_solve(
    fn: Callable[Concatenate[Tensor, ...], Tensor] | nn.Module,
    x0: Tensor,
    *,
    args: tuple[Tensor, ...] = (),
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
        args: Tensor parameters passed through to `fn`.
        maxiter: Maximum number of fixed point iterations used in both forward
            and backward solves.
        atol: Absolute tolerance for convergence.
        rtol: Relative tolerance for convergence.
    """
    t_maxiter = torch.as_tensor(maxiter, dtype=torch.int32, device=x0.device)
    t_atol = torch.as_tensor(atol, dtype=x0.dtype, device=x0.device)
    t_rtol = torch.as_tensor(rtol, dtype=x0.dtype, device=x0.device)

    with torch.no_grad():
        sol = _fixpoint_solve_impl(
            lambda z: fn(z, *args),
            x0,
            maxiter=t_maxiter,
            atol=t_atol,
            rtol=t_rtol,
        )

    x_star = fn(sol.x, *args)

    if not x_star.requires_grad:
        return x_star

    def backward_hook(g: Tensor | None, /) -> Tensor | None:
        if g is None:
            return None

        with torch.enable_grad():
            # FIXME: https://github.com/pytorch/pytorch/issues/179510
            #  cant use torch.func.vjp due to torch.compile bug.
            z0 = x_star.clone().detach().requires_grad_()
            f0 = fn(z0, *args)
            vjp_fn = lambda u: torch.autograd.grad(f0, z0, u, retain_graph=True)  # noqa: E731

        # SEC: solve u = g + (∂f/∂x)ᵀu by fixed point iteration
        # SEC: return ∂y/∂x = (∂f/∂θ)ᵀu⁎
        # FIXME: vjp_fn doesn't compose with while_loop when compiling.
        #  raises UncapturedHigherOrderOpError
        # _, vjp_fn, *_ = torch.func.vjp(lambda z: fn(z, *args), x_star)
        return _fallback_solve_impl(
            lambda u: g + vjp_fn(u)[0],
            g,
            maxiter=t_maxiter,
            atol=t_atol,
            rtol=t_rtol,
        ).x

    x_star.register_hook(backward_hook)
    return x_star
