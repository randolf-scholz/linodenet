r"""Fallback implementations of linear algebra routines."""

__all__ = [
    "SpectralNorm",
    "spectral_norm",
]

from typing import Any, Final, Optional

import torch
from torch import Tensor
from torch.linalg import vector_norm

from signatures import signature

ATOL: Final[float] = 1e-6  # 2**-23  # ~1.19e-7
RTOL: Final[float] = 1e-6  # 2**-23  # ~1.19e-7


class SpectralNorm(torch.autograd.Function):
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
    @signature("(m, n) -> ()")
    def forward(
        ctx: Any,
        A: Tensor,
        /,
        atol: float = ATOL,
        rtol: float = RTOL,
        maxiter: int = 1000,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
    ) -> Tensor:
        if A.ndim != 2:
            raise ValueError(f"Expected 2d input, got {A.shape}.")

        u = u0 if u0 is not None else A.median(dim=1).values
        v = v0 if v0 is not None else A.median(dim=0).values
        u_next = u
        v_next = v
        sigma: Tensor = torch.einsum("ij, i, j ->", A, u, v)

        for _ in range(maxiter):
            u = u_next / torch.norm(u_next)
            v = v_next / torch.norm(v_next)
            # choose optimal σ given u and v: σ = argmin ‖A - σuvᵀ‖²
            sigma = torch.einsum("ij, i, j ->", A, u, v)  # u.T @ A @ v
            # Residual: if Av = σu and Aᵀu = σv
            u_next = A @ v
            v_next = A.T @ u
            sigma_u = sigma * u
            sigma_v = sigma * v
            ru = u_next - sigma * u
            rv = v_next - sigma * v
            if (
                vector_norm(ru) <= rtol * vector_norm(sigma_u) + atol
                and vector_norm(rv) <= rtol * vector_norm(sigma_v) + atol
            ):
                break

        ctx.save_for_backward(u, v)
        return sigma

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tensor:
        u, v = ctx.saved_tensors
        return torch.einsum("..., i, j -> ...ij", grad_outputs[0], u, v)

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        r"""Jacobian-vector product forward mode."""
        u, v = ctx.saved_tensors
        return torch.einsum("...ij, i, j -> ...", grad_inputs[0], u, v)


def spectral_norm(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Tensor:
    r"""Compute the spectral norm of a matrix."""
    return SpectralNorm.apply(A, atol=atol, rtol=rtol, maxiter=maxiter, u0=u0, v0=v0)  # pyright: ignore[reportReturnType]
