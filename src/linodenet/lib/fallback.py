r"""Fallback implementations of linear algebra routines."""

__all__ = [
    "SpectralNorm",
]

from typing import Any

import torch
from torch import Tensor
from torch.linalg import vector_norm


class SpectralNorm(torch.autograd.Function):
    r"""$‖A‖_2=λ_\max(A^⊤A)$.

    The spectral norm $∥A∥_2 ≔ \sup_x ∥Ax∥_2 / ∥x∥_2$ can be shown to be equal to
    $σ_{\max}(A) = \sqrt{λ_{\max} (A^⊤A)}$, the largest singular value of $A$.

    It can be computed efficiently via Power iteration.

    One can show that the derivative is equal to:

    .. math::  \pdv{½∥A∥_2}{A} = uv^⊤

    where $u,v$ are the left/right-singular vector corresponding to $σ_\max$

    References:
        `Spectral Normalization for Generative Adversarial Networks <https://openreview.net/forum?id=B1QRgziT->`_
        Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
        `International Conference on Learning Representations 2018 <https://iclr.cc/Conferences/2018>`_
    """

    @staticmethod
    def forward(ctx: Any, /, *tensors: Tensor, **kwargs: Any) -> Tensor:
        r""".. Signature:: ``(m, n) -> 1``."""
        A = tensors[0]
        if A.ndim != 2:
            raise ValueError(f"Expected 2d input, got {A.shape}.")

        atol: float = kwargs.get("atol", 1e-6)
        rtol: float = kwargs.get("rtol", 1e-6)
        maxiter: int = kwargs.get("maxiter", 1000)
        # initialize u and v, median should be useful guess.
        u = u_next = A.median(dim=1).values
        v = v_next = A.median(dim=0).values
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
