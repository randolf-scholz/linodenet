r"""Fallback implementations of linear algebra routines."""

__all__ = [
    "State",
    "spectral_norm",
    "spectral_norm_native",
]

from typing import Final, NamedTuple, Optional

import torch
from torch import Tensor
from torch.linalg import vector_norm as v_norm

_DEFAULT_MAXITER: Final[int] = 256


class State(NamedTuple):
    r"""The iteration state of the spectral norm."""

    budget: Tensor
    u: Tensor
    v: Tensor
    g_u: Tensor
    g_v: Tensor
    A: Tensor
    atol: Tensor
    rtol: Tensor


def _cond_fn(state: State, /) -> Tensor:
    budget, u, v, _, _, A, atol, rtol = state

    # Note: important to get fresh grad_u and grad_v.
    #   for the termination check we use the simultaneous gradients.
    grad_u: Tensor
    grad_u = A.mv(v)  # gᵤ ← Av
    grad_v = A.mT.mv(u)  # gᵥ ← Aᵀu

    scale = torch.maximum(v_norm(grad_u), v_norm(grad_v))
    tol = rtol * scale + atol

    sigma_u = torch.dot(grad_u, u)  # σᵤ ← ⟨u∣gᵤ⟩
    sigma_v = torch.dot(grad_v, v)  # σᵥ ← ⟨v∣gᵥ⟩
    grad_u = grad_u.addcmul(sigma_u, u, value=-1.0)  # gᵤ ← gᵤ - σᵤu
    grad_v = grad_v.addcmul(sigma_v, v, value=-1.0)  # gᵥ ← gᵥ - σᵥv

    left_converged = v_norm(grad_u) < tol
    right_converged = v_norm(grad_v) < tol
    converged = (left_converged & right_converged).all()
    sigmas_nonnegative = (sigma_u >= 0).all() & (sigma_v >= 0).all()

    # note: we must check sigma sign because it is possible u and v are
    #   initialized such that Av=-σu and Aᵀu=-σv
    return (budget > 0) & (~converged | ~sigmas_nonnegative)


def _body_fn(state: State, /) -> State:
    budget, u, v, grad_u, grad_v, A, atol, rtol = state
    # fmt: off
    # Note: must alternate, computing both grad_u and grad_v simultaneously would be incorrect.
    # Note: unroll 8 iteration since convergence check is expensive
    grad_u = A.mv(v)             # gᵤ ← Av
    u = grad_u / v_norm(grad_u)  # u ← gᵤ/‖gᵤ‖
    grad_v = A.mT.mv(u)          # gᵥ ← Aᵀu
    v = grad_v / v_norm(grad_v)  # v ← gᵥ/‖gᵥ‖

    grad_u = A.mv(v)             # gᵤ ← Av
    u = grad_u / v_norm(grad_u)  # u ← gᵤ/‖gᵤ‖
    grad_v = A.mT.mv(u)          # gᵥ ← Aᵀu
    v = grad_v / v_norm(grad_v)  # v ← gᵥ/‖gᵥ‖

    grad_u = A.mv(v)             # gᵤ ← Av
    u = grad_u / v_norm(grad_u)  # u ← gᵤ/‖gᵤ‖
    grad_v = A.mT.mv(u)          # gᵥ ← Aᵀu
    v = grad_v / v_norm(grad_v)  # v ← gᵥ/‖gᵥ‖

    grad_u = A.mv(v)             # gᵤ ← Av
    u = grad_u / v_norm(grad_u)  # u ← gᵤ/‖gᵤ‖
    grad_v = A.mT.mv(u)          # gᵥ ← Aᵀu
    v = grad_v / v_norm(grad_v)  # v ← gᵥ/‖gᵥ‖

    grad_u = A.mv(v)             # gᵤ ← Av
    u = grad_u / v_norm(grad_u)  # u ← gᵤ/‖gᵤ‖
    grad_v = A.mT.mv(u)          # gᵥ ← Aᵀu
    v = grad_v / v_norm(grad_v)  # v ← gᵥ/‖gᵥ‖

    grad_u = A.mv(v)             # gᵤ ← Av
    u = grad_u / v_norm(grad_u)  # u ← gᵤ/‖gᵤ‖
    grad_v = A.mT.mv(u)          # gᵥ ← Aᵀu
    v = grad_v / v_norm(grad_v)  # v ← gᵥ/‖gᵥ‖

    grad_u = A.mv(v)             # gᵤ ← Av
    u = grad_u / v_norm(grad_u)  # u ← gᵤ/‖gᵤ‖
    grad_v = A.mT.mv(u)          # gᵥ ← Aᵀu
    v = grad_v / v_norm(grad_v)  # v ← gᵥ/‖gᵥ‖

    grad_u = A.mv(v)             # gᵤ ← Av
    u = grad_u / v_norm(grad_u)  # u ← gᵤ/‖gᵤ‖
    grad_v = A.mT.mv(u)          # gᵥ ← Aᵀu
    v = grad_v / v_norm(grad_v)  # v ← gᵥ/‖gᵥ‖
    # fmt: on

    return State(
        budget - 1, u, v, grad_u, grad_v, A.clone(), atol.clone(), rtol.clone()
    )


@torch.no_grad()
def _spectral_norm_forward_impl(
    A: Tensor,
    u0: Optional[Tensor],
    v0: Optional[Tensor],
    maxiter: Optional[int | Tensor],
    atol: float | Tensor,
    rtol: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    maxiter = _DEFAULT_MAXITER if maxiter is None else maxiter
    maxiter = torch.as_tensor(maxiter, device=A.device, dtype=torch.int32)
    atol = torch.as_tensor(atol, device=A.device, dtype=A.dtype)
    rtol = torch.as_tensor(rtol, device=A.device, dtype=A.dtype)

    m, n = A.shape
    u0 = u0 if u0 is not None else torch.randn(m, dtype=A.dtype, device=A.device)
    v0 = v0 if v0 is not None else torch.randn(n, dtype=A.dtype, device=A.device)
    grad_u = torch.empty_like(u0)
    grad_v = torch.empty_like(v0)

    initial_state = State(maxiter, u0, v0, grad_u, grad_v, A, atol, rtol)
    final_state = torch.while_loop(_cond_fn, _body_fn, (initial_state,))

    _, u, v, _, _, _, _, _ = final_state  # pyright: ignore[reportGeneralTypeIssues]
    sigma: Tensor = torch.einsum("ij, i, j ->", A, u, v)

    return sigma, u, v


class _SpectralNormImpl(torch.autograd.Function):
    r"""$‖A‖₂=λ_\max(AᵀA)$.

    The spectral norm $‖A‖₂ ≔ \sup_x ‖Ax‖₂ / ‖x‖₂$ can be shown to be equal to
    $σ_{\max}(A) = \sqrt{λ_{\max} (AᵀA)}$, the largest singular value of $A$.

    It can be computed efficiently via Power iteration.

    One can show that the derivative is equal to:

    .. math::  \pdv{½‖A‖₂}{A} = uvᵀ

    where $u,v$ are the left/right-singular vector corresponding to $σ_\max$

    References:
        `Spectral Normalization for Generative Adversarial Networks <https://openreview.net/forum?id=B1QRgziT->`_
        Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
        `International Conference on Learning Representations 2018 <https://iclr.cc/Conferences/2018>`_
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        A: Tensor,
        u0: Optional[Tensor],
        v0: Optional[Tensor],
        maxiter: Optional[int],
        atol: float,
        rtol: float,
        /,
    ) -> Tensor:
        sigma, u, v = _spectral_norm_forward_impl(A, u0, v0, maxiter, atol, rtol)
        ctx.save_for_backward(u, v)
        return sigma

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tensor
    ) -> tuple[Tensor, None, None, None, None, None]:
        u, v = ctx.saved_tensors
        grad = grad_outputs[0] * torch.outer(u, v)
        return grad, None, None, None, None, None


def spectral_norm(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = _DEFAULT_MAXITER,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tensor:
    r"""Compute the spectral norm of a matrix."""
    return _SpectralNormImpl.apply(A, u0, v0, maxiter, atol, rtol)  # pyright: ignore[reportReturnType]


def spectral_norm_native(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,  # noqa: ARG001
    v0: Optional[Tensor] = None,  # noqa: ARG001
    maxiter: Optional[int] = None,  # noqa: ARG001
    atol: float = 1e-6,  # noqa: ARG001
    rtol: float = 1e-6,  # noqa: ARG001
) -> Tensor:
    r"""Computes the spectral norm."""
    return torch.linalg.matrix_norm(A, ord=2)
