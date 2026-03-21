r"""Trace estimators.

Notes:
    Let vₖ = Aᵏv₀,     uₖ = (Aᵀ)ᵏv₀
    then: tr(A²ᵏ) = E[uₖᵀvₖ],  tr(A²ᵏ⁺¹) = E[uₖᵀAvₖ]
"""

__all__ = [
    "hutchinson_estimator",
    "xtrace_estimator",
    "xtrace_estimator_corrected",
    "xtrace_bilinear_estimator_experimental",
    "btrace_estimator",
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


def _project_onto_columns(basis: Tensor, vector: Tensor) -> Tensor:
    r"""Project a batched vector onto the column span of a batched basis.

    Args:
        basis: Batched orthonormal basis with shape `(..., d, k)`.
        vector: Batched vector with shape `(..., d)`.

    Returns:
        Tensor with shape `(..., d)` containing the orthogonal projection of
        `vector` onto `span(basis)`.
    """
    coeffs = torch.einsum("...dk, ...d -> ...k", basis.conj(), vector)
    return torch.einsum("...dk, ...k -> ...d", basis, coeffs)


def _normalized_inverse_h_columns(r_factor: Tensor) -> Tensor:
    r"""Return normalized columns of $(R^{-1})ᴴ$ for a square QR factor."""
    *_, rows, cols = r_factor.shape
    if rows != cols:
        raise ValueError("Efficient XTrace updates require a square R factor.")

    identity = torch.eye(cols, dtype=r_factor.dtype, device=r_factor.device)
    inverse_h = solve_triangular(r_factor.mH, identity, upper=False)
    return _normalize_columns(inverse_h)


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


    core idea:

        samples: [v₁, ..., vₖ]
        compute Qᵢ = orth(AV₋ᵢ)
        compute: trᵢ = tr(QᵢᴴAQᵢ) + vᵢᴴ(I-QᵢQᵢᴴ) A (I-QᵢQᵢᴴ)vᵢ

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


def xtrace_bilinear_estimator_experimental(
    fn: Callable[[Tensor], Tensor],
    adj_fn: Callable[[Tensor], Tensor],
    left_samples: Tensor,
    right_samples: Tensor,
) -> Tensor:
    r"""Experimental two-sided XTrace-style estimator for nonsymmetric operators.

    This estimator uses separate left and right probe families. It is based on
    the leave-one-out construction

    .. math::

       \hat tᵢ = \tr(Pᵢᴴ A Qᵢ) + uᵢᴴ(I - PᵢPᵢᴴ) A (I - QᵢQᵢᴴ) vᵢ,

    where:
        - $Qᵢ$ spans the columns of $A V_{-i}$,
        - $Pᵢ$ spans the columns of $Aᴴ U_{-i}$,
        - $vᵢ$ and $uᵢ$ are the held-out right and left probe vectors.

    The final trace estimate is the average of the leave-one-out estimates.
    This is an experimental implementation intended for moment-estimation
    experiments where left and right probe ladders are available explicitly.

    Args:
        fn: Right action of the operator, $x ↦ A x$, applied row-wise to a
            tensor with shape `(..., n, d)`.
        adj_fn: Left action of the adjoint, $x ↦ Aᴴ x$, applied row-wise to a
            tensor with shape `(..., n, d)`.
        left_samples: Left probe vectors `(..., n, d)`.
        right_samples: Right probe vectors `(..., n, d)`.

    Returns:
        Tensor with shape `(...)` containing the experimental trace estimate.
    """
    if left_samples.shape != right_samples.shape:
        raise ValueError("left_samples and right_samples must have matching shapes.")

    *_, num_samples, _ = right_samples.shape
    if num_samples == 0:
        raise ValueError("xtrace_bilinear_estimator_experimental requires samples.")

    av = fn(right_samples)  # (..., n, d), rows are A vᵢ
    ahu = adj_fn(left_samples)  # (..., n, d), rows are Aᴴ uᵢ
    estimates: list[Tensor] = []

    for i in range(num_samples):
        av_except_i = torch.cat((av[..., :i, :], av[..., i + 1 :, :]), dim=-2)
        ahu_except_i = torch.cat((ahu[..., :i, :], ahu[..., i + 1 :, :]), dim=-2)

        if num_samples == 1:
            q = right_samples.new_zeros(
                *right_samples.shape[:-2], right_samples.shape[-1], 0
            )
            p = left_samples.new_zeros(
                *left_samples.shape[:-2], left_samples.shape[-1], 0
            )
            projected_trace = right_samples.new_zeros(right_samples.shape[:-2])
        else:
            # Qᵢ = orth(A V_{-i}), shape (..., d, n-1)
            q, _ = qr(av_except_i.mT, mode="reduced")
            # Pᵢ = orth(Aᴴ U_{-i}), shape (..., d, n-1)
            p, _ = qr(ahu_except_i.mT, mode="reduced")

            # tr(Pᵢᴴ A Qᵢ), where AQᵢ is obtained by applying A to the basis columns.
            aq = fn(q.mT).mT  # (..., d, n-1)
            projected = torch.einsum("...dp, ...dq -> ...pq", p.conj(), aq)
            projected_trace = projected.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # vᵢ, uᵢ: held-out probe vectors, shape (..., d)
        v_i = right_samples[..., i, :]
        u_i = left_samples[..., i, :]

        # Right residual: (I - QᵢQᵢᴴ) vᵢ, shape (..., d)
        right_residual = v_i - _project_onto_columns(q, v_i)
        # Left residual: (I - PᵢPᵢᴴ) uᵢ, shape (..., d)
        left_residual = u_i - _project_onto_columns(p, u_i)

        # Residual correction uᵢᴴ(I - PᵢPᵢᴴ) A (I - QᵢQᵢᴴ) vᵢ
        residual_action = fn(right_residual.unsqueeze(-2)).squeeze(-2)
        residual_trace = vecdot(left_residual, residual_action, dim=-1)
        estimates.append(projected_trace + residual_trace)

    return torch.stack(estimates, dim=-1).mean(dim=-1)


def btrace_estimator(
    fn: Callable[[Tensor], Tensor],
    adj_fn: Callable[[Tensor], Tensor],
    left_samples: Tensor,
    right_samples: Tensor,
) -> Tensor:
    r"""Experimental efficient two-sided XTrace-style estimator.

    This function implements the same experimental two-sided estimator as
    `xtrace_bilinear_estimator_experimental`, but avoids recomputing leave-one-out
    QR factorizations. Instead, it uses the XTrace rank-one update identity on
    both sides:

    .. math::

       QᵢQᵢᴴ = Q(I - sᵢsᵢᴴ)Qᴴ, \qquad PᵢPᵢᴴ = P(I - tᵢtᵢᴴ)Pᴴ,

    where the columns `sᵢ` and `tᵢ` are obtained from the normalized columns of
    `(R_Q^{-1})ᴴ` and `(R_P^{-1})ᴴ`, respectively.

    The implemented estimator uses the projector form

    .. math::

       \hat tᵢ = \tr(Πᴸᵢ A Πᴿᵢ)
              + uᵢᴴ(I - Πᴸᵢ) A (I - Πᴿᵢ) vᵢ,

    with `Πᴿᵢ = QᵢQᵢᴴ` and `Πᴸᵢ = PᵢPᵢᴴ`.

    Notes:
        This remains an experimental generalization. It is intended as an
        efficient implementation vehicle for further moment-estimation work.
    """
    if left_samples.shape != right_samples.shape:
        raise ValueError("left_samples and right_samples must have matching shapes.")

    *_, num_samples, dim = right_samples.shape
    if num_samples == 0:
        raise ValueError(
            "xtrace_bilinear_estimator_efficient_experimental requires samples."
        )
    if num_samples > dim:
        raise ValueError(
            "Efficient bilinear XTrace currently requires num_samples <= dimension."
        )

    # Right sketch: V columns are the probe vectors, Y = A V.
    # v_cols, av_cols: (..., d, n)
    v_cols = right_samples.mT
    av_cols = fn(right_samples).mT
    # Q: (..., d, n), R_q: (..., n, n)
    q, r_q = qr(av_cols, mode="reduced")
    # S columns are the normalized null-space update vectors sᵢ.
    # s: (..., n, n)
    s = _normalized_inverse_h_columns(r_q)

    # Left sketch: U columns are the probe vectors, AᴴU drives the left basis.
    # u_cols, ahu_cols: (..., d, n)
    u_cols = left_samples.mT
    ahu_cols = adj_fn(left_samples).mT
    # P: (..., d, n), R_p: (..., n, n)
    p, r_p = qr(ahu_cols, mode="reduced")
    # T columns are the normalized update vectors tᵢ for the left basis.
    # t: (..., n, n)
    t = _normalized_inverse_h_columns(r_p)

    # H = Pᴴ A Q, C = Qᴴ P. Shapes: (..., n, n)
    aq = fn(q.mT).mT
    h = torch.einsum("...dp, ...dq -> ...pq", p.conj(), aq)
    c = torch.einsum("...dq, ...dp -> ...qp", q.conj(), p)

    # Projected trace term:
    # tr(Πᴸᵢ A Πᴿᵢ) = tr((I - tᵢtᵢᴴ) H (I - sᵢsᵢᴴ) C)
    #               = tr(HC) - tᵢᴴHC tᵢ - sᵢᴴCH sᵢ + (tᵢᴴHsᵢ)(sᵢᴴCtᵢ)
    hc = h @ c
    ch = c @ h
    trace_hc = hc.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    s_cols = s.mT  # (..., n, n), row i contains sᵢᴴ data as a vector
    t_cols = t.mT  # (..., n, n), row i contains tᵢᴴ data as a vector
    d_t_hc_t = torch.einsum("...ni, ...ij, ...nj -> ...n", t_cols.conj(), hc, t_cols)
    d_s_ch_s = torch.einsum("...ni, ...ij, ...nj -> ...n", s_cols.conj(), ch, s_cols)
    d_t_h_s = torch.einsum("...ni, ...ij, ...nj -> ...n", t_cols.conj(), h, s_cols)
    d_s_c_t = torch.einsum("...ni, ...ij, ...nj -> ...n", s_cols.conj(), c, t_cols)
    projected_trace = trace_hc - d_t_hc_t - d_s_ch_s + d_t_h_s * d_s_c_t

    # W = QᴴV and Z = PᴴU collect probe coordinates in the full left/right bases.
    # Shapes: (..., n, n)
    w = torch.einsum("...dq, ...dn -> ...qn", q.conj(), v_cols)
    z = torch.einsum("...dp, ...dn -> ...pn", p.conj(), u_cols)
    w_rows = w.mT  # (..., n, n), row i is wᵢ
    z_rows = z.mT  # (..., n, n), row i is zᵢ

    # xᵢ = wᵢ - <sᵢ, wᵢ> sᵢ,  yᵢ = zᵢ - <tᵢ, zᵢ> tᵢ
    alpha = vecdot(s_cols, w_rows, dim=-1)
    beta = vecdot(t_cols, z_rows, dim=-1)
    x = w_rows - alpha.unsqueeze(-1) * s_cols
    y = z_rows - beta.unsqueeze(-1) * t_cols

    # Residual vectors:
    # (I - Πᴿᵢ) vᵢ = vᵢ - Q xᵢ,   (I - Πᴸᵢ) uᵢ = uᵢ - P yᵢ
    right_residuals = right_samples - torch.einsum("...dq, ...nq -> ...nd", q, x)
    left_residuals = left_samples - torch.einsum("...dp, ...np -> ...nd", p, y)

    # Residual correction uᵢᴴ(I - Πᴸᵢ) A (I - Πᴿᵢ) vᵢ, all samples at once.
    residual_actions = fn(right_residuals)
    residual_trace = vecdot(left_residuals, residual_actions, dim=-1)

    return (projected_trace + residual_trace).mean(dim=-1)
