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
