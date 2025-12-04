# Filters & Cells

Recall the Kalman update equations:

$$\begin{aligned}
    μ' &= μ - Σ Hᵀ (H Σ Hᵀ + R)⁻¹ (H μ - y)
\\  Hμ' &= Hμ - H Σ Hᵀ (H Σ Hᵀ + R)⁻¹ (H μ - y)
\end{aligned}$$

Assuming H is surjective, then $R=HQHᵀ$ for some $Q$. In particular,

$$(H Σ Hᵀ + R)⁻¹ = (H (Σ + Q) Hᵀ)⁻¹ = H⁺ (Σ + Q)⁻¹ (Hᵀ)⁺$$

if $H=𝕀$, this simplifies to:

$$\begin{aligned}
    μ' &= μ - Σ (Σ + R)⁻¹ (μ - y)
\end{aligned}$$

Consider the case when $R=λ𝕀$, in particular $R$ commutes with $Σ$, thus:

- Kalmanesque Filter: $x' = x ±
