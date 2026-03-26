r"""Fallback implementations of singular triplet routines."""

__all__ = [
    "singular_triplet",
    "singular_triplet_native",
]

from typing import Any, Optional

import torch
from torch import Tensor

from .spectral_norm import _DEFAULT_MAXITER, _spectral_norm_forward_impl


class _SingularTripletImpl(torch.autograd.Function):
    r"""Compute the dominant singular triplet via power iteration."""

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx: Any,
        A: Tensor,
        u0: Optional[Tensor],
        v0: Optional[Tensor],
        maxiter: int,
        atol: float,
        rtol: float,
        /,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sigma, u, v = _spectral_norm_forward_impl(A, u0, v0, maxiter, atol, rtol)
        ctx.save_for_backward(A, sigma, u, v)
        return sigma, u, v

    @staticmethod
    def backward(
        ctx, *outer: Tensor | None
    ) -> tuple[Tensor, None, None, None, None, None]:
        A, sigma, u, v = ctx.saved_tensors
        g_sigma, g_u, g_v = outer

        if g_sigma is None:
            g_sigma = torch.zeros((), dtype=A.dtype, device=A.device)
        if g_u is None:
            g_u = torch.zeros_like(u)
        if g_v is None:
            g_v = torch.zeros_like(v)

        g_sigma_out = g_sigma * torch.outer(u, v)

        if not (g_u.any().item() or g_v.any().item()):
            return g_sigma_out, None, None, None, None, None

        m, n = A.shape
        options = {"dtype": A.dtype, "device": A.device}
        zero_u = torch.zeros((m, 1), **options)
        zero_v = torch.zeros((n, 1), **options)
        eye_m = torch.eye(m, **options)
        eye_n = torch.eye(n, **options)

        k_top = torch.cat((sigma * eye_m, -A, u.unsqueeze(-1), zero_u), dim=1)
        k_bottom = torch.cat((-A.T, sigma * eye_n, zero_v, v.unsqueeze(-1)), dim=1)
        k_mat = torch.cat((k_top, k_bottom), dim=0)
        c_vec = torch.cat((g_u, g_v), dim=0)

        x = torch.linalg.lstsq(k_mat, c_vec).solution
        p = x[:m]
        q = x[m : m + n]

        g_u_out = torch.outer(p - torch.dot(u, p) * u, v)
        g_v_out = torch.outer(u, q - torch.dot(v, q) * v)
        return g_sigma_out + g_u_out + g_v_out, None, None, None, None, None


def singular_triplet(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,
    v0: Optional[Tensor] = None,
    maxiter: Optional[int] = _DEFAULT_MAXITER,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute the dominant singular triplet of a matrix."""
    return _SingularTripletImpl.apply(A, u0, v0, maxiter, atol, rtol)


def singular_triplet_native(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,  # noqa: ARG001
    v0: Optional[Tensor] = None,  # noqa: ARG001
    maxiter: Optional[int] = None,  # noqa: ARG001
    atol: float = 1e-6,  # noqa: ARG001
    rtol: float = 1e-6,  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    U, S, Vh = torch.linalg.svd(A)
    # cols of U = LSV, rows of Vh: RSV
    return S[0], U[:, 0], Vh[0, :]
