r"""Fallback implementations of singular triplet routines."""

__all__ = [
    "ATOL",
    "RTOL",
    "SingularTriplet",
    "singular_triplet",
    "singular_triplet_native",
]

from typing import Any, Final, Optional, Protocol

import torch
from torch import Tensor
from torch.linalg import vector_norm

from signatures import signature

ATOL: Final[float] = 1e-6  # 2**-23  # ~1.19e-7
RTOL: Final[float] = 1e-6  # 2**-23  # ~1.19e-7


class SingularTriplet(Protocol):
    r"""Protocol for singular triplet implementations."""

    @signature("(m, n) -> [(), (m), (n)]")
    def __call__(
        self,
        A: Tensor,
        /,
        *,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
        maxiter: Optional[int] = None,
        atol: float = ATOL,
        rtol: float = RTOL,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Computes the singular triplet.

        Args:
            A: The input matrix (shape: M×N).
            u0: The initial guess for the left singular vector (shape: M).
            v0: The initial guess for the right singular vector (shape: N).
            maxiter: The maximum number of iterations. (Default: O(M+N))
            atol: The absolute tolerance. (Default: 1e-6)
            rtol: The relative tolerance. (Default: 1e-6)

        Returns:
            sigma: The singular value (scaler).
            u: The left singular vector (shape: M).
            v: The right singular vector (shape: N).
        """
        ...


class _SingularTripletImpl(torch.autograd.Function):
    r"""Compute the dominant singular triplet via power iteration."""

    @staticmethod
    def forward(
        ctx: Any,
        A: Tensor,
        /,
        atol: float = ATOL,
        rtol: float = RTOL,
        maxiter: Optional[int] = None,
        u0: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if A.ndim != 2:
            raise ValueError(f"Expected 2d input, got {A.shape}.")
        if not A.is_floating_point():
            raise TypeError("Expected a floating point tensor.")

        m, n = A.shape
        maxiter = maxiter if maxiter is not None else (m + n + 64)

        u = u0 if u0 is not None else A.median(dim=1).values
        v = v0 if v0 is not None else A.median(dim=0).values
        u_next = u
        v_next = v
        sigma: Tensor = torch.einsum("ij, i, j ->", A, u, v)

        for _ in range(maxiter):
            u = u_next / torch.norm(u_next)
            v = v_next / torch.norm(v_next)
            sigma = torch.einsum("ij, i, j ->", A, u, v)
            u_next = A @ v
            v_next = A.T @ u
            sigma_u = sigma * u
            sigma_v = sigma * v
            ru = u_next - sigma_u
            rv = v_next - sigma_v
            if (
                vector_norm(ru) <= rtol * vector_norm(sigma_u) + atol
                and vector_norm(rv) <= rtol * vector_norm(sigma_v) + atol
            ):
                break

        if (not torch.isfinite(sigma).item()) or (sigma <= 0).item():
            raise RuntimeError(
                "Computation resulted in invalid singular value. "
                "Try increasing maxiter or tolerance."
            )

        ctx.save_for_backward(A, sigma, u, v)
        return sigma, u, v

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Optional[Tensor]
    ) -> tuple[Tensor, None, None, None, None, None]:
        A, sigma, u, v = ctx.saved_tensors
        g_sigma, g_u, g_v = grad_outputs

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
    maxiter: Optional[int] = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute the dominant singular triplet of a matrix."""
    return _SingularTripletImpl.apply(  # pyright: ignore[reportReturnType]
        A, atol=atol, rtol=rtol, maxiter=maxiter, u0=u0, v0=v0
    )


def singular_triplet_native(
    A: Tensor,
    /,
    *,
    u0: Optional[Tensor] = None,  # noqa: ARG001
    v0: Optional[Tensor] = None,  # noqa: ARG001
    maxiter: Optional[int] = None,  # noqa: ARG001
    atol: float = 1e-8,  # noqa: ARG001
    rtol: float = 1e-5,  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Computes the singular triplet."""
    U, S, Vh = torch.linalg.svd(A)
    # cols of U = LSV, rows of Vh: RSV
    return S[0], U[:, 0], Vh[0, :]
