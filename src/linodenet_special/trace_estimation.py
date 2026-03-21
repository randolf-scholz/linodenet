r"""Trace estimators.

Notes:
    Let vₖ = Aᵏv₀,     uₖ = (Aᵀ)ᵏv₀
    then: tr(A²ᵏ) = E[uₖᵀvₖ],  tr(A²ᵏ⁺¹) = E[uₖᵀAvₖ]
"""

__all__ = ["hutchinson_estimator", "xtrace_estimator"]

from collections.abc import Callable

import torch
from torch import Tensor
from torch.linalg import qr, solve_triangular, vecdot, vector_norm

from signatures import signature


@signature("(..., n, d) -> (...)")
def hutchinson_estimator(fn: Callable[[Tensor], Tensor], samples: Tensor) -> Tensor:
    r"""Estimate the trace of a matrix with Hutchinson's estimator.

    .. math:: tr(A) = E[vᵀAv], where E[v]=0 and Cov[v]= 𝕀

    Args:
        fn: Matrix-vector product function, i.e. $x ↦ Ax$ (batched).
        samples: Random samples to use for the estimator.
            Shape: `(..., n, d)`, with `...` batch size, `n` number of samples,
            and `d` dimension.

    Returns:
        Tensor: The estimated trace.
    """
    return vecdot(samples, fn(samples), dim=-1).mean(dim=-1)


@signature("(..., n, d) -> (...)")
def xtrace_estimator(fn: Callable[[Tensor], Tensor], samples: Tensor) -> Tensor:
    r"""Estimate the trace of a matric.

    Args:
        fn: matrix-vector product function, i.e. x ↦ Ax (batched)
        samples: random samples to use for the estimator.
            shape: (..., n, d), with `...` batch size, n: num_samples, d: dimension.

    Returns:
        Tensor: The estimated trace.
    """
    V = samples.mT  # (..., d, n)
    *_, d, n = V.shape
    k = min(n, d)
    Y = fn(V.mT).mT  # (..., d, n)
    Q, R = qr(Y, mode="reduced")  # (..., d, k), (..., k, n)
    Z = fn(Q.mT).mT  # (..., d, k)
    H = torch.einsum("...kd, ...dj -> ...kj", Q.mH, Z)  # (..., k, k)
    W = torch.einsum("...kd, ...dn -> ...nk", Q.mH, V)  # (..., n, k)
    T = torch.einsum("...kd, ...dn -> ...nk", Z.mH, V)  # (..., n, k)

    # Note: compute S=R⁻¹ ⟺ S R = Iₖ  (or: R S = Iₙ)
    I = torch.eye(k, dtype=samples.dtype, device=samples.device)
    S = solve_triangular(I, R.mH, upper=True, left=False)  # (..., n, k)
    S = S / vector_norm(S, dim=-2, keepdim=True)  # (..., n, k)

    # compute xᵢ = wᵢ - ⟨sᵢ∣wᵢ⟩ sᵢ
    X = W - torch.einsum("...nk, ...nk, ...nl -> ...nl", S.conj(), W, S)  # (..., n, k)
    # compute tr_i = ⟨xᵢ|H|xᵢ⟩ - ⟨sᵢ|H|sᵢ⟩ + ⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ⟩ - ⟨tᵢ∣xᵢ⟩
    TRS = (
        torch.einsum("...nk, ...kl, ...nl -> ...n", X.conj(), H, X)  # ⟨xᵢ|H|xᵢ⟩
        - torch.einsum("...nk, ...kl, ...nl -> ...n", S.conj(), H, S)  # - ⟨sᵢ|H|sᵢ⟩
        - torch.einsum("...nk, ...nk -> ...n", T.conj(), X)  # - ⟨tᵢ∣xᵢ⟩
        + (
            torch.einsum("...nk, ...nk -> ...n", W.conj(), S)  # ⟨wᵢ∣sᵢ⟩
            * torch.einsum("...nk, ...kn -> ...n", S.conj(), R)  # ⟨sᵢ∣rᵢ⟩
        )
    )
    # compute tr = tr(H) + mean(tr_i)
    return H.diagonal(dim1=-2, dim2=-1).sum(dim=-1) + TRS.mean(dim=-1)
