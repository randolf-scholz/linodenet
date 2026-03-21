r"""Trace estimators.

Notes:
    Let vₖ = Aᵏv₀,     uₖ = (Aᵀ)ᵏv₀
    then: tr(A²ᵏ) = E[uₖᵀvₖ],  tr(A²ᵏ⁺¹) = E[uₖᵀAvₖ]
"""

__all__ = [
    "hutchinson_estimator",
    "xtrace_estimator",
    "xtrace_estimator_corrected",
    "naive_estimator",
]

from collections.abc import Callable

import torch
from torch import Tensor
from torch.linalg import qr, solve_triangular, vecdot, vector_norm

from signatures import signature


def _normalize_columns(matrix: Tensor) -> Tensor:
    r"""Normalize the columns of a batched matrix."""
    return matrix / vector_norm(matrix, dim=-2, keepdim=True)


def _diag_prod(lhs: Tensor, rhs: Tensor) -> Tensor:
    r"""Return diag(lhsᴴ rhs) for batched matrices with matching shape."""
    return torch.einsum("...dm, ...dm -> ...m", lhs.conj(), rhs)


@signature("(..., n, d) -> (...)")
def naive_estimator(fn: Callable[[Tensor], Tensor], samples: Tensor) -> Tensor:
    r"""Estimate the trace of a matric, realizing the full matrix."""
    I = torch.eye(samples.shape[-1], dtype=samples.dtype, device=samples.device)
    A = fn(I)
    return torch.einsum("...dd -> ...", A)


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

    Algorithm:
        1: Draw Ω ∼ Unif{±1}^{N×m/2}
        2: Y ← AΩ
        3: (Q, R) ← qr(Y, ’econ’)
        4: Z ← AQ
        5: H ← QᴴZ, W ← QᴴΩ, T ← ZᴴΩ
        6: S ← R⁻ᴴ
        7: S ← S · diag(∥sᵢ∥: i=1…m/2)
        8: for i = 1 … m/2 do
        9:     xᵢ ← wᵢ − ⟨sᵢ∣wᵢ⟩·sᵢ
        10:    trᵢ ← tr(H) − ⟨sᵢ|H sᵢ⟩ + ⟨wᵢ∣sᵢ⟩·⟨sᵢ∣rᵢ⟩ − ⟨tᵢ|xᵢ⟩ + ⟨xᵢ|Hxᵢ⟩
        11: end for
        12: tr ← mean(trᵢ: i=1…m/2)
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


@signature("(..., n, d) -> (...)")
def xtrace_estimator_corrected(
    fn: Callable[[Tensor], Tensor], samples: Tensor
) -> Tensor:
    r"""Estimate the trace of a matrix using the original XTrace MATLAB algorithm.

    This is a direct Torch transcription of the reference MATLAB code. The input
    `samples` stores row-wise probe vectors with shape `(..., n, d)`. The
    original MATLAB algorithm is written for a column-wise probe matrix
    $Ω ∈ ℝᵈˣᵐ$, so we transpose into column form internally and mirror the
    MATLAB algebra closely.

    Notes:
        The MATLAB reference takes a matvec budget `m_budget` and internally
        uses `m = floor(m_budget / 2)` probe vectors. This function instead
        follows the local API convention that `samples` already contains the
        probe vectors, so all `n` rows are consumed directly.
    """
    *_, m, d = samples.shape
    if m == 0:
        raise ValueError("xtrace_estimator_corrected requires at least one sample.")

    # MATLAB: Om = sqrt(N) * cnormc(randn(N, m))
    # Here we reuse the provided probes as the m columns of Ω.
    # Omega: (..., d, m)
    omega = d**0.5 * _normalize_columns(samples.mT)

    # MATLAB: Y = A * Om
    # Y: (..., d, m)
    y = fn(omega.mT).mT
    # MATLAB: [Q, R] = qr(Y, 0)
    # Q: (..., d, m), R: (..., m, m)
    q, r = qr(y, mode="reduced")

    # MATLAB: W = Q' * Om
    # W: (..., m, m)
    w = torch.einsum("...dm, ...dn -> ...mn", q.conj(), omega)

    # MATLAB: S = cnormc(inv(R)')
    # S: (..., m, m), columns of (R^{-1})ᴴ normalized to unit norm.
    identity = torch.eye(m, dtype=samples.dtype, device=samples.device)
    s = solve_triangular(r.mH, identity, upper=False)
    s = _normalize_columns(s)

    # MATLAB:
    # scale = (N - m + 1) ./ (N - ||w_i||² + |<s_i, w_i> ||s_i|| |²)
    # column norms / diagonal products: (..., m)
    w_norm_sq = vector_norm(w, dim=-2).square()
    s_norm = vector_norm(s, dim=-2)
    d_sw = _diag_prod(s, w)
    scale = (d - m + 1) / (d - w_norm_sq + (d_sw * s_norm).abs().square())

    # MATLAB: Z = A * Q
    # Z: (..., d, m)
    z = fn(q.mT).mT
    # MATLAB: H = Q' * Z
    # H: (..., m, m)
    h = torch.einsum("...dm, ...dn -> ...mn", q.conj(), z)
    # MATLAB: HW = H * W
    # HW: (..., m, m)
    hw = h @ w
    # MATLAB: T = Z' * Om
    # T: (..., m, m)
    t = torch.einsum("...dm, ...dn -> ...mn", z.conj(), omega)

    # Column-wise diagonal contractions used by the estimator correction terms.
    # All shapes below are (..., m).
    d_shs = _diag_prod(s, h @ s)
    d_tw = _diag_prod(t, w)
    d_whw = _diag_prod(w, hw)
    d_s_r_minus_hw = _diag_prod(s, r - hw)
    d_t_minus_hhw_s = _diag_prod(t - h.mH @ w, s)

    # MATLAB:
    # ests_i = tr(H)
    #        - <s_i, H s_i>
    #        + ( <w_i, H w_i> - <t_i, w_i>
    #            + <t_i - H' w_i, s_i><s_i, w_i>
    #            + |<s_i, w_i>|² <s_i, H s_i>
    #            + conj(<s_i, w_i>) <s_i, r_i - H w_i> ) * scale_i
    trace_h = h.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    ests = (
        trace_h
        - d_shs
        + (
            d_whw
            - d_tw
            + d_t_minus_hhw_s * d_sw
            + d_sw.abs().square() * d_shs
            + d_sw.conj() * d_s_r_minus_hw
        )
        * scale
    )

    # MATLAB: t = mean(ests)
    return ests.mean(dim=-1)
