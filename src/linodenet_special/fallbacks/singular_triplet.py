r"""Fallback implementations of singular triplet routines."""

__all__ = [
    "singular_triplet",
    "singular_triplet_native",
]

from typing import Optional

import torch
from torch import Tensor
from torch.linalg import vecdot, vector_norm

from linodenet_special.interfaces import DEFAULT_SPECTRAL_NORM_MAXITER

from .spectral_norm import _spectral_norm_forward_impl


class _SingularTripletImpl(torch.autograd.Function):
    r"""Compute the dominant singular triplet via power iteration."""

    @staticmethod
    @torch.no_grad()
    def forward(
        A: Tensor,
        u0: Tensor,
        v0: Tensor,
        maxiter: int,
        atol: float,
        rtol: float,
        /,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _spectral_norm_forward_impl(A, u0, v0, maxiter, atol, rtol)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.save_for_backward(inputs[0], *output)  # A, σ, u, v

    @staticmethod
    def backward(
        ctx, *outer: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
        A, sigma, u, v = ctx.saved_tensors
        g_sigma, g_u, g_v = outer

        zero_u = torch.zeros_like(u)
        zero_v = torch.zeros_like(v)

        if g_sigma is None:
            g_sigma = torch.zeros((), dtype=A.dtype, device=A.device)
        if g_u is None:
            g_u = torch.zeros_like(u)
        if g_v is None:
            g_v = torch.zeros_like(v)

        g_sigma_out = g_sigma * torch.outer(u, v)

        if not (g_u.any().item() or g_v.any().item()):
            return g_sigma_out, zero_u, zero_v, None, None, None

        m, n = A.shape
        eye_m = torch.eye(m, dtype=A.dtype, device=A.device)
        eye_n = torch.eye(n, dtype=A.dtype, device=A.device)

        k_top = torch.cat(
            [sigma * eye_m, -A, u.unsqueeze(-1), zero_u.unsqueeze(-1)], dim=1
        )
        k_bottom = torch.cat(
            [-A.T, sigma * eye_n, zero_v.unsqueeze(-1), v.unsqueeze(-1)], dim=1
        )
        k_mat = torch.cat((k_top, k_bottom), dim=0)
        c_vec = torch.cat((g_u, g_v), dim=0)

        x = torch.linalg.lstsq(k_mat, c_vec).solution
        p = x[:m]
        q = x[m : m + n]

        g_u_out = torch.outer(p - vecdot(u, p) * u, v)
        g_v_out = torch.outer(u, q - vecdot(v, q) * v)
        return g_sigma_out + g_u_out + g_v_out, zero_u, zero_v, None, None, None


def singular_triplet(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: int | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute the dominant singular triplet of a matrix."""
    maxiter = DEFAULT_SPECTRAL_NORM_MAXITER[A.dtype] if maxiter is None else maxiter
    u = (
        u0.detach().clone()
        if u0 is not None
        else torch.randn(A.shape[-2], dtype=A.dtype, device=A.device)
    )
    v = (
        v0.detach().clone()
        if v0 is not None
        else torch.randn(A.shape[-1], dtype=A.dtype, device=A.device)
    )
    u = u / vector_norm(u, dim=-1, keepdims=True)
    v = v / vector_norm(v, dim=-1, keepdims=True)
    return _SingularTripletImpl.apply(A, u, v, maxiter, atol, rtol)


def singular_triplet_native(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,  # noqa: ARG001
    v0: Optional[Tensor] = None,  # noqa: ARG001
    maxiter: int | None = None,  # noqa: ARG001
    atol: float = 1e-6,  # noqa: ARG001
    rtol: float = 1e-6,  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    U, S, Vh = torch.linalg.svd(A)
    # cols of U = LSV, rows of Vh: RSV
    return S[0], U[:, 0], Vh[0, :]
