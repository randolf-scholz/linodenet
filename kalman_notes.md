# Kalman Notes

## Index

- [Kalman Notes](#kalman-notes)
  - [Index](#index)
  - [Setup](#setup)
  - [From Constraint to Penalty](#from-constraint-to-penalty)
  - [Density View](#density-view)
  - [Bayesian Specialization](#bayesian-specialization)
  - [Recovering the Kalman Filter](#recovering-the-kalman-filter)
    - [Newton-Step View of the Mean Update](#newton-step-view-of-the-mean-update)
    - [Comparing Euclidean, Natural, Newton, and Proximal Updates](#comparing-euclidean-natural-newton-and-proximal-updates)
    - [Positive-Definite Geometry and Safe Covariance Updates](#positive-definite-geometry-and-safe-covariance-updates)
    - [One-Step Online Updates and Tempered Kalman](#one-step-online-updates-and-tempered-kalman)
  - [What Is `λ` for the Kalman Filter?](#what-is-λ-for-the-kalman-filter)
  - [Observation-Space View](#observation-space-view)

## Setup

Let

- latent state: $x$
- latent distribution family: $p(x ∣ θ)$
- observation map: $y = h(x)$
- induced observation distribution: $p_θ(y) = h_* p(x ∣ θ)$
- previous parameter: $θ_{\text{old}}$
- observed information in observation space: $q(y)$

The basic problem is:

> update $θ_{\text{old}}$ to a new parameter $θ'$ so that the predicted
> observation law $p_θ(y)$ agrees better with the observed law $q(y)$,
> while changing the latent distribution as little as possible.

## From Constraint to Penalty

The cleanest formulation is a constrained projection:

```math
\begin{aligned}
θ' &= \arg\min_θ D_{\text{latent}} \bigl(p(x ∣ θ), p(x ∣ θ_{\text{old}})\bigr) \\
  &\text{subject to } p_θ(y) = q(y)
\end{aligned}
```

This says:

- in latent space, stay as close as possible to the old belief
- in observation space, match the new information exactly

In practice, exact matching may be impossible, ill-posed, or undesirable because
the observation itself is noisy. Then the hard constraint is relaxed into a soft
penalty:

```math
\begin{aligned}
θ' &= \arg\min_θ D_{\text{obs}} \bigl(q(y), p_θ(y)\bigr)
  + λ D_{\text{latent}} \bigl(p(x ∣ θ), p(x ∣ θ_{\text{old}})\bigr)
\end{aligned}
```

This is the penalized or Lagrangian form of the constrained problem:

- $D_{\text{obs}}$ measures mismatch in observation space
- $D_{\text{latent}}$ measures how far the latent law moves from the previous one
- $λ > 0$ trades off data fit against trust in the previous latent belief

Interpretation:

- $λ \to 0$: fit the observed distribution almost regardless of latent drift
- $λ \to \infty$: barely move the latent distribution

## Density View

If the observation is a single point $y_{\text{obs}}$, then $q(y)$ is the Dirac mass
$∆_{y_{\text{obs}}}$. In that case minimizing observation mismatch is equivalent to
maximizing predictive likelihood:

```math
\begin{aligned}
θ' &= \arg\min_θ \bigl(-\log p_θ(y_{\text{obs}})\bigr)
  + λ D_{\text{latent}} \bigl(p(x ∣ θ), p(x ∣ θ_{\text{old}})\bigr)
\end{aligned}
```

This is the same tradeoff seen from densities:

- the first term rewards parameters that make the observation likely
- the second term penalizes moving too far from the old latent belief

## Bayesian Specialization

The fully Bayesian update is not just a point update for $θ$; it updates the
whole latent distribution. Let $r(x)$ be a candidate posterior distribution.
Then Bayes' rule gives

```math
\begin{aligned}
p(x ∣ y_{\text{obs}}) ∝ p(y_{\text{obs}} ∣ x) p_{\text{old}}(x),
\end{aligned}
```

where $p_{\text{old}}(x)$ is the previous latent belief. The posterior is equivalently
the minimizer of the free-energy objective

```math
\begin{aligned}
r &= \arg\min_{r} D_{𝖪𝖫}\bigl(r(x) ∥ p_{\text{old}}(x)\bigr)
- 𝐄_r[\log p(y_{\text{obs}} ∣ x)]
\end{aligned}
```

This is exactly the same pattern as observation-fit plus latent-regularization.

The first term keeps $r$ close to the old belief, and the second rewards
distributions that explain the observation well.

If we restrict $r$ to a parametrized family $r(x) = p(x ∣ θ)$, we get

```math
\begin{aligned}
θ' &= \arg\min_θ
D_{𝖪𝖫}\bigl(p(x ∣ θ) ∥ p(x ∣ θ_{\text{old}})\bigr)
- 𝐄_{p(x ∣ θ)}[\log p(y_{\text{obs}} ∣ x)]
\end{aligned}
```

This is the principled Bayesian version of

```math
\begin{aligned}
θ' &= \arg\min_θ
D_{\text{obs}}\bigl(q(y), p_θ(y)\bigr)
+ λ D_{\text{latent}} \bigl(p(x ∣ θ), p(x ∣ θ_{\text{old}})\bigr)
\end{aligned}
```

## Recovering the Kalman Filter

Take the linear-Gaussian model

```math
\begin{aligned}
x &∼ 𝓝(μ, Σ), \\
y ∣ x &∼ 𝓝(Hx, R)
\end{aligned}
```

Let the candidate updated latent law be

```math
\begin{aligned}
p(x ∣ θ) &= 𝓝(μ', Σ'), \\
θ &= (μ', Σ'), \\
θ_{\text{old}} &= (μ, Σ)
\end{aligned}
```

The Bayesian objective becomes

```math
\begin{aligned}
J(μ', Σ')
&= D_{𝖪𝖫}\bigl(𝓝(μ', Σ') ∥ 𝓝(μ, Σ)\bigr) \\
&\quad - 𝐄_{𝓝(μ', Σ')} \bigl[\log 𝓝(y_{\text{obs}}; Hx, R)\bigr]
\end{aligned}
```

Up to constants independent of $(μ', Σ')$,

```math
\begin{aligned}
J(μ', Σ')
&= D_{𝖪𝖫}\bigl(𝓝(μ', Σ') ∥ 𝓝(μ, Σ)\bigr) \\
&\quad + ½
𝐄\bigl[(y_{\text{obs}} - Hx)ᵀ R⁻¹ (y_{\text{obs}} - Hx)\bigr]
\end{aligned}
```

Since $Hx ∼ 𝓝(Hμ', HΣ'Hᵀ)$, the expectation is

```math
\begin{aligned}
𝐄\bigl[(y_{\text{obs}} - Hx)ᵀ R⁻¹ (y_{\text{obs}} - Hx)\bigr]
&= (y_{\text{obs}} - Hμ')ᵀ R⁻¹ (y_{\text{obs}} - Hμ') \\
&\quad + \operatorname{tr}(R⁻¹ HΣ'Hᵀ)
\end{aligned}
```

Therefore

```math
\begin{aligned}
J(μ', Σ')
&= D_{𝖪𝖫}\bigl(𝓝(μ', Σ') ∥ 𝓝(μ, Σ)\bigr) \\
&\quad + ½(y_{\text{obs}} - Hμ')ᵀ R⁻¹ (y_{\text{obs}} - Hμ') \\
&\quad + ½\operatorname{tr}(R⁻¹ HΣ'Hᵀ)

+ \text{const}
  \end{aligned}
```

Minimizing this objective yields the unique Gaussian posterior

```math
\begin{aligned}
Σ' &= (Σ⁻¹ + Hᵀ R⁻¹ H)⁻¹, \\
μ' &= Σ'(Σ⁻¹μ + Hᵀ R⁻¹ y_{\text{obs}})
\end{aligned}
```

Using the Woodbury identity, this is equivalent to the standard Kalman form

```math
\begin{aligned}
K &= Σ Hᵀ(HΣ Hᵀ + R)⁻¹, \\
μ' &= μ + K(y_{\text{obs}} - Hμ), \\
Σ' &= (I - KH)Σ
\end{aligned}
```

So the classical Kalman filter is exactly the solution of the
observation-fit plus latent-regularization objective when:

- the latent family is Gaussian
- the observation model is linear-Gaussian
- the observation is a point measurement with Gaussian noise
- the latent regularizer is KL to the previous latent Gaussian
- the observation-fit term is the expected negative log-likelihood

### Newton-Step View of the Mean Update

There is a useful special case in which the Kalman update is exactly a single
Newton step.

If we optimize only over the latent mean, or equivalently over the latent state
point estimate $x$, the negative log posterior is

```math
\begin{aligned}
L(x)
&= ½(x - μ)ᵀ Σ⁻¹(x - μ) \\
&\quad + ½(y_{\text{obs}} - Hx)ᵀ R⁻¹(y_{\text{obs}} - Hx)
\end{aligned}
```

This is a quadratic function of $x$. Its gradient and Hessian are

```math
\begin{aligned}
∇ L(x) &= Σ⁻¹(x - μ) + Hᵀ R⁻¹(Hx - y_{\text{obs}}), \\
∇²L(x) &= Σ⁻¹ + Hᵀ R⁻¹ H
\end{aligned}
```

The Hessian is constant, so one Newton step from any starting point $x₀$
reaches the exact minimizer:

```math
\begin{aligned}
x₁
&= x₀ - (∇² L)⁻¹∇ L(x₀) \\
&= \arg\min_x L(x)
\end{aligned}
```

Starting specifically from the prior mean $x₀ = μ$,

```math
\begin{aligned}
x₁
&= μ - (Σ⁻¹ + Hᵀ R⁻¹ H)⁻¹Hᵀ R⁻¹(Hμ - y_{\text{obs}}) \\
&= μ + K(y_{\text{obs}} - Hμ),
\end{aligned}
```

with

```math
\begin{aligned}
K = Σ Hᵀ(HΣ Hᵀ + R)⁻¹
\end{aligned}
```

So the classical Kalman mean update is exactly one Newton step on the quadratic
negative log posterior.

This also clarifies what it is not:

- it is not, in general, a plain gradient step with a scalar step size
- it is a preconditioned gradient step with matrix preconditioner
  $(Σ⁻¹ + Hᵀ R⁻¹ H)⁻¹$
- equivalently, it is one exact Newton step because the objective is quadratic

The covariance update is different. The full objective in $(μ', Σ')$ is not
jointly quadratic because it contains $-\log\det Σ'$ through the Gaussian KL.
Its minimizer is still available in closed form,

```math
\begin{aligned}
Σ'⁻¹ = Σ⁻¹ + Hᵀ R⁻¹ H,
\end{aligned}
```

but that part should be understood as the exact Gaussian/Bayesian covariance
update, not merely as "one Newton step on a quadratic objective."

### Comparing Euclidean, Natural, Newton, and Proximal Updates

This is the linear-Gaussian specialization of the generic Bayesian/free-energy
objective from above:

```math
\begin{aligned}
θ' &= \arg\min_θ
D_{𝖪𝖫}\bigl(p(x ∣ θ) ∥ p(x ∣ θ_{\text{old}})\bigr)
- 𝐄_{p(x ∣ θ)}[\log p(y_{\text{obs}} ∣ x)].
\end{aligned}
```

So the comparison below keeps the same observation-fit plus
latent-regularization functional and changes only the local update rule. Up to
constants, write

```math
\begin{aligned}
J(μ', Σ')
&= ½(μ' - μ)ᵀ Σ⁻¹ (μ' - μ) \\
&\quad + ½(y_{\text{obs}} - Hμ')ᵀ R⁻¹(y_{\text{obs}} - Hμ') \\
&\quad + ½\operatorname{tr}\bigl((Σ⁻¹ + HᵀR⁻¹H)Σ'\bigr) \\
&\quad - ½\log\det Σ'.
\end{aligned}
```

Define

```math
\begin{aligned}
r &≔ y_{\text{obs}} - Hμ, \\
A &≔ HᵀR⁻¹H, \\
K &≔ ΣHᵀ(HΣHᵀ + R)⁻¹.
\end{aligned}
```

At the current point $(μ, Σ)$, the gradients are

```math
\begin{aligned}
∇_μ J &= -HᵀR⁻¹r, \\
∇_Σ J &= ½A.
\end{aligned}
```

For the Gaussian family, the Fisher / KL metric is

```math
\begin{aligned}
G_μ &= Σ⁻¹, \\
G_Σ^{-1}[U] &= 2ΣUΣ.
\end{aligned}
```

The resulting updates are:

| formula | mean update | covariance update |
| --- | --- | --- |
| Euclidean gradient: $θ₊ = θ - α∇J(θ)$ | $μ₊ = μ + αHᵀR⁻¹r$ | $Σ₊ = Σ - ½αA$ |
| Natural gradient: $θ₊ = θ - αG(θ)^{-1}∇J(θ)$ | $μ₊ = μ + αΣHᵀR⁻¹r$ | $Σ₊ = Σ - αΣAΣ$ |
| Block Newton, with Gauss-Newton on the mean block: $θ₊ = θ - [∇²J(θ)]^{-1}∇J(θ)$ | $μ₊ = μ + (Σ⁻¹ + A)⁻¹HᵀR⁻¹r = μ + Kr$ | $Σ₊ = Σ - ΣAΣ$ |
| Exact KL-proximal / Bayes: $θ₊ = \arg\min J(θ)$ | $μ₊ = μ + Kr$ | $Σ₊ = (Σ⁻¹ + A)⁻¹ = (I - KH)Σ$ |

A few points are worth emphasizing:

- the Euclidean updates ignore latent geometry and the covariance step need not
  preserve positive definiteness
- the natural-gradient and block-Newton covariance steps coincide here because
  the Hessian of $-\log\det Σ'$ at the current point is exactly the Gaussian
  Fisher / KL metric
- the additive covariance update $Σ - ΣAΣ$ is only the first-order
  approximation of the exact Kalman covariance

```math
\begin{aligned}
(Σ⁻¹ + A)⁻¹
= Σ - ΣAΣ + O(A²)
\end{aligned}
```

- so the Kalman mean update is a single Newton step, but the Kalman covariance
  update is best understood as the exact KL-proximal / Bayesian solution rather
  than as a plain additive gradient step

### Positive-Definite Geometry and Safe Covariance Updates

The covariance lives in the open cone of symmetric positive-definite matrices
$𝕊_{++}^n$, not in a flat vector space. At any $Σ ∈ 𝕊_{++}^n$, the tangent space is

```math
\begin{aligned}
T_Σ 𝕊_{++}^n = \operatorname{Sym}(n).
\end{aligned}
```

So the issue is not how to get a tangent direction, but how to map that tangent
direction back into $𝕊_{++}^n$ without leaving the cone.

Reuse the covariance directions from the previous comparison:

```math
\begin{aligned}
U_{\text{Euc}}(α) &≔ -½αA, \\
U_{\text{Nat}}(α) &≔ -αΣAΣ, \\
U_{\text{Newt}} &≔ -ΣAΣ,
\end{aligned}
```

where $A = HᵀR⁻¹H ⪰ 0$.

If these directions are used additively in covariance coordinates, then

```math
\begin{aligned}
Σ₊ = Σ + U
\end{aligned}
```

stays positive definite only locally:

| raw step | covariance update | SPD guarantee |
| --- | --- | --- |
| Euclidean | $Σ₊ = Σ - ½αA$ | only if $I - ½αΣ^{-1/2}AΣ^{-1/2} \succ 0$ |
| Natural | $Σ₊ = Σ - αΣAΣ$ | only if $I - αΣ^{1/2}AΣ^{1/2} \succ 0$ |
| Newton | $Σ₊ = Σ - ΣAΣ$ | same condition as the natural step with $α = 1$ |
| Exact proximal / Kalman | $Σ₊ = (Σ⁻¹ + αA)⁻¹$ | always, for every $α ≥ 0$ |

An intrinsic way to stay on $𝕊_{++}^n$ is to use the affine-invariant
Riemannian exponential map

```math
\begin{aligned}
\operatorname{Exp}_Σ(U)
= Σ^{1/2}\exp\!\bigl(Σ^{-1/2}UΣ^{-1/2}\bigr)Σ^{1/2}.
\end{aligned}
```

Another way is to reparametrize $Σ$ and do the update in coordinates that map
back into $𝕊_{++}^n$ by construction.

The table below keeps the same underlying covariance direction
$U ∈ \{U_{\text{Euc}}(α), U_{\text{Nat}}(α), U_{\text{Newt}}\}$ and changes only
how that direction is realized while preserving positive definiteness.

It is important to distinguish three different notions that are often conflated:

- a generic factor $B$ with $Σ = BBᵀ$
- a Cholesky factor $L$ with $Σ = LLᵀ$, where $L$ is lower triangular with
  positive diagonal
- the principal square root $Σ^{1/2}$, which is itself symmetric positive
  definite and therefore lives on $𝕊_{++}^n$ again

The factor column below uses the first notion, not the principal square root.

| update schema | intrinsic / no parametrization | precision $Λ = Σ⁻¹$ | factor $Σ = BBᵀ$ | Cholesky $Σ = LLᵀ$ | log-matrix $Σ = \exp(S)$ |
| --- | --- | --- | --- | --- | --- |
| Euclidean | $Σ₊ = \operatorname{Exp}_Σ(U_{\text{Euc}}(α)) = Σ^{1/2}\exp(-½αΣ^{-1/2}AΣ^{-1/2})Σ^{1/2}$ | $Λ₊ = Λ + ½αΛAΛ$ | $B₊ = B\exp(-¼αB^{-1}AB^{-T})$ | $L₊ = L\,\operatorname{chol}\!\bigl(\exp(-½αL^{-1}AL^{-T})\bigr)$ | $S₊ = S - ½αD\log_Σ[A]$ |
| Natural | $Σ₊ = \operatorname{Exp}_Σ(U_{\text{Nat}}(α)) = Σ^{1/2}\exp(-αΣ^{1/2}AΣ^{1/2})Σ^{1/2}$ | $Λ₊ = Λ + αA$ | $B₊ = B\exp(-½αBᵀAB)$ | $L₊ = L\,\operatorname{chol}\!\bigl(\exp(-αLᵀAL)\bigr)$ | $S₊ = S - αD\log_Σ[ΣAΣ]$ |
| Newton | $Σ₊ = \operatorname{Exp}_Σ(U_{\text{Newt}}) = Σ^{1/2}\exp(-Σ^{1/2}AΣ^{1/2})Σ^{1/2}$ | $Λ₊ = Λ + A$ | $B₊ = B\exp(-½BᵀAB)$ | $L₊ = L\,\operatorname{chol}\!\bigl(\exp(-LᵀAL)\bigr)$ | $S₊ = S - D\log_Σ[ΣAΣ]$ |
| Exact | $Σ₊ = (Σ⁻¹ + A)⁻¹ = (I - KH)Σ$ | $Λ₊ = Λ + A$ | $B₊ = B(I + BᵀAB)^{-1/2}$ | $L₊ = L\,\operatorname{chol}\!\bigl((I + LᵀAL)^{-1}\bigr)$ | $S₊ = -\log(\exp(-S) + A)$ |

Here $D\log_Σ$ denotes the Fréchet derivative of the matrix logarithm, i.e. the
inverse of the Fréchet derivative of the matrix exponential at $S = \log Σ$:

```math
\begin{aligned}
D\log_Σ[U] = (D\exp_S)^{-1}[U].
\end{aligned}
```

Also, $C^{-1/2}$ denotes the principal inverse square root of an SPD matrix
$C \succ 0$, and $\operatorname{chol}(C)$ denotes its lower-triangular Cholesky
factor.

A few consequences are especially useful:

- the intrinsic column uses the same tangent direction as before, but replaces
  the unsafe additive step by the Riemannian exponential map
- the precision column is particularly natural because
  $ΔΣ = -Σ(ΔΛ)Σ + O(ΔΛ²)$, so the natural and Newton covariance directions become
  additive precision updates
- in particular, the Newton row in precision coordinates gives
  $Λ₊ = Λ + A$ and $Σ₊ = (Λ + A)⁻¹$, which is exactly the classical Kalman
  covariance update. A short derivation is:
  $(Λ + ΔΛ)^{-1} = Λ^{-1} - Λ^{-1}(ΔΛ)Λ^{-1} + O(ΔΛ²)
  = Σ - Σ(ΔΛ)Σ + O(ΔΛ²)$.
  Since the Newton covariance direction in $Σ$-coordinates is
  $ΔΣ_{\text{Newt}} = -ΣAΣ$, matching first-order terms gives $ΔΛ = A$, so the
  Newton covariance step becomes $Λ₊ = Λ + A$. For the linear-Gaussian model
  this is not merely a local approximation: the exact Bayesian posterior
  already satisfies $Σ₊^{-1} = Σ^{-1} + A = Λ + A$. So the precision/Newton row
  matches Kalman because precision is the natural information parameter and the
  exact posterior update is affine in that parameter, not because the
  covariance objective becomes quadratic in $Λ$.
- the factor, Cholesky, and log-matrix columns preserve positive definiteness by
  construction
- if one insists on the principal square-root parametrization
  $P = Σ^{1/2} \in 𝕊_{++}^n$, then $P$ itself is another SPD variable and must be
  updated with its own SPD-preserving geometry; there is no equally simple
  free-factor formula because the constraint is no longer just invertibility

### One-Step Online Updates and Tempered Kalman

For a general model, solving the full variational problem at every time step may
be too expensive. A natural alternative is to do a single local update.

The important distinction is between:

- a plain Euclidean gradient step in parameter space
- a KL-proximal or mirror-descent step in distribution space

These are not the same.

Start from the generic online objective

```math
\begin{aligned}
Jₜ(θ)
&= D_{\text{obs}}\bigl(qₜ(y), p_θ(y)\bigr)
+ λ D_{\text{latent}}\bigl(p(x ∣ θ), p(x ∣ θₜ)\bigr)
\end{aligned}
```

If we linearize the observation term around the current parameter $θₜ$, we get

```math
\begin{aligned}
D_{\text{obs}}\bigl(qₜ(y), p_θ(y)\bigr) &≈ \text{const} + gₜᵀ(θ - θₜ),
\end{aligned}
```

with

```math
\begin{aligned}
gₜ  &= ∇_θ D_{\text{obs}}\bigl(qₜ(y), p_θ(y)\bigr)\big|_{θ = θₜ}
\end{aligned}
```

If we also use the second-order expansion of the latent divergence around
$θₜ$, then because the divergence has a minimum at $θₜ$,

```math
\begin{aligned}
D_{\text{latent}}\bigl(p(x ∣ θ), p(x ∣ θₜ)\bigr)
&≈ ½(θ - θₜ)ᵀ Gₜ(θ - θₜ),
\end{aligned}
```

where

```math
\begin{aligned}
Gₜ  &= ∇_θ² D_{\text{latent}}\bigl(p(x ∣ θ), p(x ∣ θₜ)\bigr)\big|_{θ = θₜ}
\end{aligned}
```

So the local model of the objective is

```math
\begin{aligned}
Jₜ(θ)
&≈ \text{const}
+ gₜᵀ(θ - θₜ)
+ ½λ(θ - θₜ)ᵀ Gₜ(θ - θₜ)
\end{aligned}
```

Differentiating and setting the gradient to zero gives

```math
\begin{aligned}
gₜ + λ Gₜ(θ - θₜ) = 0,
\end{aligned}
```

so

```math
\begin{aligned}
∆θₜ ≔ θ_{t+1} - θₜ = -\frac{1}{λ} Gₜ⁻¹ gₜ
\end{aligned}
```

Writing the step size as $ηₜ = 1/λ$, this becomes

```math
\begin{aligned}
∆θₜ = -ηₜ Gₜ⁻¹ gₜ
\end{aligned}
```

This is the matrix-preconditioned local update behind mirror descent, natural
gradient, and proximal filtering. The choice of $Gₜ$ determines the geometry:

- $Gₜ = I$ gives an ordinary Euclidean gradient step
- $Gₜ =$ local Hessian gives a Newton step
- $Gₜ =$ local Fisher / KL metric gives a natural-gradient step

For the variational Gaussian objective from above,

```math
\begin{aligned}
J(μ', Σ')
&= D_{𝖪𝖫}\bigl(𝓝(μ', Σ') ∥ 𝓝(μ, Σ)\bigr) \\
&\quad - 𝐄_{𝓝(μ', Σ')}[\log p(y_{\text{obs}} ∣ x)],
\end{aligned}
```

a plain Euclidean gradient step in $(μ', Σ')$ taken from the current point
$(μ, Σ)$ uses

```math
\begin{aligned}
∇_μ J \big|_{(μ,Σ)}
&= Hᵀ R⁻¹(Hμ - y_{\text{obs}}), \\
∇_Σ J \big|_{(μ,Σ)} &= ½ Hᵀ R⁻¹H
\end{aligned}
```

So the explicit gradient step with learning rate $α$ is

```math
\begin{aligned}
μ_{\text{next}}
&= μ + α Hᵀ R⁻¹(y_{\text{obs}} - Hμ), \\
Σ_{\text{next}}
&= Σ - ½αHᵀ R⁻¹H
\end{aligned}
```

This is not the Kalman update. In particular:

- the covariance update is additive in covariance coordinates
- it need not preserve positive definiteness
- it ignores the curvature encoded by the prior covariance $Σ$

The more principled one-step update is the KL-proximal problem

```math
\begin{aligned}
q_{\text{next}}
&= \arg\min_q
𝐄_q[-\log p(y_{\text{obs}} ∣ x)]

+ \frac{1}{η}D_{𝖪𝖫}(q ∥ q_{\text{old}}),
  \end{aligned}
```

where $q_{\text{old}}$ is the current latent belief and $η > 0$ is the step size.

Taking first variations gives

```math
\begin{aligned}
\log q_{\text{next}}(x)
&= \log q_{\text{old}}(x) + η \log p(y_{\text{obs}} ∣ x) + \text{const},
\end{aligned}
```

so

```math
\begin{aligned}
q_{\text{next}}(x) ∝ q_{\text{old}}(x) p(y_{\text{obs}} ∣ x)^η
\end{aligned}
```

This is a tempered Bayesian update. For $η = 1$, it is exact Bayes. For
$η < 1$, it is a partial update. For $η > 1$, it overweights the observation.

Now specialize to the linear-Gaussian model

```math
\begin{aligned}
q_{\text{old}}(x) &= 𝓝(μ, Σ), \\
y_{\text{obs}} ∣ x &∼ 𝓝(Hx, R)
\end{aligned}
```

Then $q_{\text{next}}$ is again Gaussian, and multiplying the exponents shows that its
natural parameters update additively:

```math
\begin{aligned}
Σ_{\text{next}}⁻¹
&= Σ⁻¹ + η Hᵀ R⁻¹ H, \\
Σ_{\text{next}}⁻¹μ_{\text{next}}
&= Σ⁻¹μ + η Hᵀ R⁻¹ y_{\text{obs}}
\end{aligned}
```

Equivalently,

```math
\begin{aligned}
μ_{\text{next}}
&= Σ_{\text{next}}(Σ⁻¹μ + η Hᵀ R⁻¹ y_{\text{obs}}),
\end{aligned}
```

and, using Woodbury,

```math
\begin{aligned}
K_η &= Σ Hᵀ(HΣ Hᵀ + R/η)⁻¹, \\
μ_{\text{next}} &= μ + K_η(y_{\text{obs}} - Hμ), \\
Σ_{\text{next}} &= (I - K_η H)Σ
\end{aligned}
```

This is exactly a tempered Kalman filter:

- $η = 1$ gives the classical Kalman update
- $η < 1$ is equivalent to using a larger effective observation noise $R/η$
- $η > 1$ is equivalent to using a smaller effective observation noise

So if the question is "what one-step update for $Σ$ matches Kalman?", the answer
is

```math
\begin{aligned}
Σ_{\text{next}}⁻¹ = Σ⁻¹ + η Hᵀ R⁻¹ H,
\end{aligned}
```

or equivalently

```math
\begin{aligned}
Σ_{\text{next}} = (Σ⁻¹ + η Hᵀ R⁻¹ H)⁻¹
\end{aligned}
```

At $η = 1$, this is exactly the Kalman covariance update. By contrast, the
plain Euclidean gradient step produces

```math
\begin{aligned}
Σ_{\text{next}} = Σ - ½αHᵀ R⁻¹H,
\end{aligned}
```

which is generally not Kalman.

## What Is `λ` for the Kalman Filter?

For the exact Bayesian/Kalman update, $λ$ is not a free tuning parameter.
It is effectively

```math
\begin{aligned}
λ = 1
\end{aligned}
```

This holds provided both terms are written in the same units, namely nats:

- $D_{\text{latent}} = D_{𝖪𝖫}(p(x ∣ θ) ∥ p(x ∣ θ_{\text{old}}))$
- $D_{\text{obs}} = -𝐄_{p(x ∣ θ)}[\log p(y_{\text{obs}} ∣ x)]$
  or an equivalent observation-space negative log-density term

Why $λ = 1$:

- Bayes' rule multiplies prior and likelihood
- taking $-\log$ turns that product into a sum
- both terms enter with coefficient $1$

Any other $λ$ corresponds to a tempered or heuristic update rather than the
classical Kalman filter. In that sense:

- $λ = 1$ gives the standard Bayesian/Kalman update
- $λ ≠ 1$ reweights prior vs. data and no longer matches the exact posterior

## Observation-Space View

Define the predictive observation moments

```math
\begin{aligned}
μ_y &= Hμ, \\
Σ_y &= HΣ Hᵀ + R
\end{aligned}
```

The Kalman posterior induces

```math
\begin{aligned}
μ_y' &= Hμ', \\
Σ_y' &= HΣ'Hᵀ + R
\end{aligned}
```

These can be written purely in observation-space quantities as

```math
\begin{aligned}
μ_y'
&= μ_y + (Σ_y - R)Σ_y⁻¹(y_{\text{obs}} - μ_y) \\
&= y_{\text{obs}} - RΣ_y⁻¹(y_{\text{obs}} - μ_y), \\
Σ_y'
&= Σ_y - (Σ_y - R)Σ_y⁻¹(Σ_y - R)
\end{aligned}
```

These formulas describe the observation-space posterior induced by the latent
Kalman update. In general they do not determine the latent posterior uniquely
unless the decoder map is sufficiently invertible.
