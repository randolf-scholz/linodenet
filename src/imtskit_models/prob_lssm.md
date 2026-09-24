# Probabilistic Latent State-Space Model Interface

## Index

- [Goal](#goal)
- [Core Interface](#core-interface)
  - [State Conventions](#state-conventions)
  - [Minimal Protocols](#minimal-protocols)
  - [Mathematical Signatures](#mathematical-signatures)
  - [Internal Decoder Factorizations](#internal-decoder-factorizations)
    - [Parameter-space decoder](#parameter-space-decoder)
    - [Data-space decoder](#data-space-decoder)
- [Kalman Filter as an Instance of the Interface](#kalman-filter-as-an-instance-of-the-interface)
  - [1. Propagation (Kalman)](#1-propagation-kalman)
  - [2. Decoder (Kalman)](#2-decoder-kalman)
  - [3. Update (Kalman)](#3-update-kalman)
- [Extended Kalman Filter as an Instance of the Interface](#extended-kalman-filter-as-an-instance-of-the-interface)
  - [1. Propagation (EKF)](#1-propagation-ekf)
  - [2. Decoder (EKF)](#2-decoder-ekf)
  - [3. Update (EKF)](#3-update-ekf)
- [A General Tractable Gaussian Variant](#a-general-tractable-gaussian-variant)
  - [Latent Family](#latent-family)
  - [1. Propagation from a Linear ODE/SDE](#1-propagation-from-a-linear-odesde)
  - [2. Neural Decoder in Belief Space](#2-neural-decoder-in-belief-space)
    - [Gaussian observation head](#gaussian-observation-head)
    - [Normalizing-flow observation head](#normalizing-flow-observation-head)
  - [3. Gradient-Based Update](#3-gradient-based-update)
    - [Euclidean gradient step](#euclidean-gradient-step)
    - [Metric-aware step](#metric-aware-step)
- [History-Conditioned Path Forecasting](#history-conditioned-path-forecasting)
  - [How History Enters](#how-history-enters)
  - [Path-Forecasting Interface](#path-forecasting-interface)
  - [Exact Gaussian Path Law for Linear ODE/SDE Dynamics](#exact-gaussian-path-law-for-linear-odesde-dynamics)
  - [Decoding the Latent Path](#decoding-the-latent-path)
    - [Kalman path decoder](#kalman-path-decoder)
    - [Neural path decoder](#neural-path-decoder)
  - [Filtering Versus Smoothing](#filtering-versus-smoothing)
- [Summary Table](#summary-table)
- [Recommended Interpretation](#recommended-interpretation)

## Goal

We want an abstraction for probabilistic state-space models that operates on
the **latent belief** rather than on a single latent state value.

At each time `t`, let

- `X_t` be the latent random state,
- `x_t` be a point in latent space, i.e. a realized latent value,
- `θ_t` be the parameter of the maintained latent distribution,
- `p(· ∣ θ_t)` be the latent belief density/distribution of `X_t`,
- `Y_t` be the observation random variable,
- `y_t` be an observed value.

The filter carries `θ_t`, not a point estimate `x_t`.

We reserve `p(x ∣ θ)` for the density evaluated at the point `x`, whereas
`p(· ∣ θ)` denotes the full density/distribution.

The model is decomposed into three components:

1. **Propagation**: evolve the latent belief forward in time.
2. **Decoder**: map the latent belief to a predictive observation law.
3. **Update**: incorporate new observations into the latent belief.

This covers the Kalman filter, EKF, and more general belief-space models.

## Core Interface

### State Conventions

Use the usual prediction/update notation:

- posterior belief after assimilating data up to time `t`:
  `p(· ∣ θ_t⁺)`
- predicted prior before assimilating the next observation:
  `p(· ∣ θ_{t+Δ}⁻)`

The filtering loop is

```text
θ_{t+Δ}⁻ = propagate(Δ, θ_t⁺)
q_{t+Δ}⁻ = decode(θ_{t+Δ}⁻)
θ_{t+Δ}⁺ = update(y_{t+Δ}, θ_{t+Δ}⁻, decoder)
```

Here `q_{t+Δ}⁻` is the predictive observation distribution induced by the
decoder from the latent belief.

### Minimal Protocols

The abstraction is intentionally belief-centric. The decoder does not need to
expose a latent-state map `x ↦ y`; it only needs to expose the predictive
observation law and its log-density.

```python
from typing import Protocol, TypeVar

Delta = TypeVar("Delta")
Theta = TypeVar("Theta")
Observation = TypeVar("Observation")
ObsLaw = TypeVar("ObsLaw")
Tensor = TypeVar("Tensor")


class ObservationLaw(Protocol[Observation]):
    def log_prob(self, y: Observation, /) -> Tensor: ...


class Propagation(Protocol[Delta, Theta]):
    def __call__(self, delta: Delta, theta: Theta, /) -> Theta: ...


class Decoder(Protocol[Theta, Observation, ObsLaw]):
    def __call__(self, theta: Theta, /) -> ObsLaw: ...
    def log_prob(self, y: Observation, theta: Theta, /) -> Tensor: ...


class Update(Protocol[Theta, Observation, ObsLaw]):
    def __call__(
        self,
        y: Observation,
        theta_pred: Theta,
        decoder: Decoder[Theta, Observation, ObsLaw],
        /,
    ) -> Theta: ...
```

### Mathematical Signatures

The same interface can be written more compactly as

```text
propagate : Δ × Θ → Θ
decode    : Θ → 𝒫(𝒴)
update    : 𝒴 × Θ × Decoder → Θ
```

with the contract

```text
decode(theta).log_prob(y) = decoder.log_prob(y, theta).
```

### Internal Decoder Factorizations

The external decoder interface stays the same, but internally it may be
implemented in either of two common ways.

#### Parameter-space decoder

```text
ω = g(θ)
q(· ∣ ω) = decode(θ)
```

Then

```text
decoder.log_prob(y, θ) = log q(y ∣ g(θ)).
```

This is the natural choice for belief-space models and is the most tractable
online.

#### Data-space decoder

```text
q_θ(y) = p(ϕ⁻¹(y) ∣ θ) |det Dϕ⁻¹(y)|
```

Then

```text
decoder.log_prob(y, θ) = log q_θ(y).
```

This is the pushforward/flow view. It still fits the same external decoder
signature, even though internally it uses an invertible map in data space.

## Kalman Filter as an Instance of the Interface

Take the linear-Gaussian model

```math
\begin{aligned}
X_{k+1} &= A_k X_k + b_k + W_k, \qquad W_k ∼ 𝓝(0, Q_k), \\
Y_k &= H_k X_k + c_k + V_k, \qquad V_k ∼ 𝓝(0, R_k).
\end{aligned}
```

The latent family is Gaussian:

```text
θ = (μ, Σ),     p(· ∣ θ) = 𝓝(·; μ, Σ).
```

### 1. Propagation (Kalman)

Signature:

```text
propagate(Δ, (μ, Σ)) -> (μ⁻, Σ⁻)
```

Discrete-time Kalman choice:

```math
\begin{aligned}
μ⁻ &= A_k μ + b_k, \\
Σ⁻ &= A_k Σ A_kᵀ + Q_k.
\end{aligned}
```

So the propagator is an exact Gaussian moment update.

### 2. Decoder (Kalman)

Signature:

```text
decode((μ, Σ)) -> q(·) = 𝓝(·; m, S)
```

Kalman choice:

```math
\begin{aligned}
ω &= (m, S), \\
m &= H_k μ + c_k, \\
S &= H_k Σ H_kᵀ + R_k, \\
q(· ∣ ω) &= 𝓝(·; m, S).
\end{aligned}
```

Hence

```math
\begin{aligned}
\log q(y_{\mathrm{obs}} ∣ g(θ))
= \log 𝓝(y_{\mathrm{obs}}; H_k μ + c_k, H_k Σ H_kᵀ + R_k).
\end{aligned}
```

This is a parameter-space decoder with `g(μ, Σ) = (H_k μ + c_k, H_k Σ H_kᵀ + R_k)`.

### 3. Update (Kalman)

Signature:

```text
update(y_obs, (μ⁻, Σ⁻), decoder) -> (μ⁺, Σ⁺)
```

Kalman choice:

```math
\begin{aligned}
K_k &= Σ⁻ H_kᵀ(H_k Σ⁻ H_kᵀ + R_k)^{-1}, \\
μ⁺ &= μ⁻ + K_k(y_{\mathrm{obs}} - H_k μ⁻ - c_k), \\
Σ⁺ &= (I - K_k H_k) Σ⁻.
\end{aligned}
```

In numerically sensitive settings, the Joseph form is preferable:

```math
\begin{aligned}
Σ⁺
= (I - K_k H_k) Σ⁻ (I - K_k H_k)ᵀ + K_k R_k K_kᵀ.
\end{aligned}
```

So the classical Kalman filter fits the interface with:

- exact Gaussian propagation,
- exact Gaussian predictive decoder,
- exact Gaussian conditioning update.

## Extended Kalman Filter as an Instance of the Interface

Take the nonlinear state-space model

```math
\begin{aligned}
X_{k+1} &= f_k(X_k) + W_k, \qquad W_k ∼ 𝓝(0, Q_k), \\
Y_k &= h_k(X_k) + V_k, \qquad V_k ∼ 𝓝(0, R_k).
\end{aligned}
```

Again use the Gaussian latent family

```text
θ = (μ, Σ),     p(· ∣ θ) = 𝓝(·; μ, Σ).
```

### 1. Propagation (EKF)

Signature:

```text
propagate(Δ, (μ, Σ)) -> (μ⁻, Σ⁻)
```

EKF choice:

```math
\begin{aligned}
μ⁻ &= f_k(μ), \\
F_k &= \left.\frac{∂f_k}{∂x}\right|_{x=μ}, \\
Σ⁻ &= F_k Σ F_kᵀ + Q_k.
\end{aligned}
```

This is a locally Gaussian closure of the nonlinear propagation.

### 2. Decoder (EKF)

Signature:

```text
decode((μ, Σ)) -> q(·) = 𝓝(·; m, S)
```

EKF choice:

```math
\begin{aligned}
H_k &= \left.\frac{∂h_k}{∂x}\right|_{x=μ}, \\
m &= h_k(μ), \\
S &= H_k Σ H_kᵀ + R_k, \\
q(·) &= 𝓝(·; m, S).
\end{aligned}
```

So the EKF decoder is still a belief-to-observation map, but only after local
linearization of the nonlinear measurement model.

### 3. Update (EKF)

Signature:

```text
update(y_obs, (μ⁻, Σ⁻), decoder) -> (μ⁺, Σ⁺)
```

EKF choice:

```math
\begin{aligned}
H_k &= \left.\frac{∂h_k}{∂x}\right|_{x=μ⁻}, \\
K_k &= Σ⁻ H_kᵀ(H_k Σ⁻ H_kᵀ + R_k)^{-1}, \\
μ⁺ &= μ⁻ + K_k(y_{\mathrm{obs}} - h_k(μ⁻)), \\
Σ⁺ &= (I - K_k H_k) Σ⁻ (I - K_k H_k)ᵀ + K_k R_k K_kᵀ.
\end{aligned}
```

So the EKF fits the same interface as the Kalman filter:

- the latent family is still Gaussian,
- propagation and decoding are approximate Gaussian closures,
- the update is Kalman conditioning on the linearized local observation model.

## A General Tractable Gaussian Variant

We now want a variant that is still online-tractable but more expressive.

Requirements:

- latent belief is Gaussian,
- propagation comes from a linear ODE/SDE,
- decoder is neural,
- update is a gradient-based or proximal step,
- all three components obey the same interface above.

The key design choice is that the neural decoder should be a **belief-space
decoder** `θ ↦ q(· ∣ g(θ))`, not a latent-conditioned model `p(· ∣ x)`. This
avoids any runtime expectation over `x`.

### Latent Family

Let

```text
θ = (μ, Σ),     p(· ∣ θ) = 𝓝(·; μ, Σ).
```

### 1. Propagation from a Linear ODE/SDE

Use the linear continuous-time model

```math
\begin{aligned}
dX_t = A_t X_t\,dt + a_t\,dt + G_t\,dW_t.
\end{aligned}
```

Over a step of length `Δ`, the Gaussian family is preserved. The propagator
still has the same signature

```text
propagate(Δ, (μ, Σ)) -> (μ⁻, Σ⁻).
```

In the time-homogeneous case,

```math
\begin{aligned}
Φ_Δ &= \exp(AΔ), \\
b_Δ &= \int_0^Δ \exp(Aτ)a\,dτ, \\
Q_Δ &= \int_0^Δ \exp(Aτ)GGᵀ\exp(Aᵀτ)\,dτ,
\end{aligned}
```

and

```math
\begin{aligned}
μ⁻ &= Φ_Δ μ + b_Δ, \\
Σ⁻ &= Φ_Δ Σ Φ_Δᵀ + Q_Δ.
\end{aligned}
```

So propagation is exact and closed-form at the level of Gaussian moments.

### 2. Neural Decoder in Belief Space

Use a neural map

```math
\begin{aligned}
ω = g_ψ(θ),
\end{aligned}
```

where `g_ψ` acts on Gaussian belief parameters, for example on `(μ, Σ)` or on a
stable reparametrization such as `(μ, L)` with `Σ = LLᵀ`.

The decoder signature remains

```text
decode((μ, Σ)) -> q_ψ(· ∣ ω = g_ψ(μ, Σ)).
```

Two tractable decoder heads are especially natural.

#### Gaussian observation head

Let `g_ψ(θ) = (m_ψ(θ), S_ψ(θ))`, with `S_ψ(θ) ≻ 0`. Then

```math
\begin{aligned}
q_ψ(· ∣ g_ψ(θ)) = 𝓝(·; m_ψ(θ), S_ψ(θ)).
\end{aligned}
```

This is the simplest tractable choice.

#### Normalizing-flow observation head

Let `ω = g_ψ(θ)` parameterize an invertible flow in observation space:

```math
\begin{aligned}
z &∼ p_0(z), \\
y &= ϕ_ψ(z; ω).
\end{aligned}
```

Then

```math
\begin{aligned}
\log q_ψ(y ∣ g_ψ(θ))
= \log p_0(ϕ_ψ^{-1}(y; ω))
+ \log\left|\det D_y ϕ_ψ^{-1}(y; ω)\right|.
\end{aligned}
```

This remains tractable online because the flow is conditioned on `θ`, not on the
latent sample `x`. No integral over `x` appears at runtime.

### 3. Gradient-Based Update

The updater keeps the same signature

```text
update(y_obs, θ_pred, decoder) -> θ_post.
```

A general tractable choice is to define the objective

```math
\begin{aligned}
J(θ; θ⁻, y_{\mathrm{obs}})
&= λ\,D_{𝖪𝖫}\bigl(p(·∣θ)\,\|\,p(·∣θ⁻)\bigr) \\
&\quad - \log q_ψ\bigl(y_{\mathrm{obs}} ∣ g_ψ(θ)\bigr),
\end{aligned}
```

and then perform one or a few optimization steps starting from `θ⁻`.

Here `θ⁻` is the **predicted prior for the current assimilation step**, held
fixed during the inner optimization. To avoid confusing the filter time index
with the optimizer iteration index, one may write the inner iterates as
`ϑ^{(0)}, ϑ^{(1)}, …` with `ϑ^{(0)} = θ⁻` and `θ⁺ = ϑ^{(K)}`.

#### Euclidean gradient step

```math
\begin{aligned}
θ_{0} &= θ⁻, \\
θ_{k+1} &= θ_k - η ∇_θ J(θ_k; θ⁻, y_{\mathrm{obs}}).
\end{aligned}
```

If only a **single explicit Euclidean step** is taken from the initialization
`θ₀ = θ⁻`, then the KL anchor does not contribute at first order:

```math
\begin{aligned}
θ_1
&= θ⁻ - η ∇_θ J(θ⁻; θ⁻, y_{\mathrm{obs}}) \\
&= θ⁻ + η ∇_θ \log q_ψ(y_{\mathrm{obs}} ∣ g_ψ(θ))\big|_{θ=θ⁻},
\end{aligned}
```

because `θ ↦ D_{𝖪𝖫}(p(·∣θ)\,\|\,p(·∣θ⁻))` attains its minimum at `θ = θ⁻`, so

```math
\begin{aligned}
∇_θ D_{𝖪𝖫}(p(·∣θ)\,\|\,p(·∣θ⁻))\big|_{θ=θ⁻} = 0.
\end{aligned}
```

Thus a one-step Euclidean update is driven only by the likelihood term. The KL
regularizer enters only through local curvature, so it matters once one takes
multiple inner steps or uses a proximal / natural-gradient style update.

#### Metric-aware step

A better general form is

```math
\begin{aligned}
θ_{k+1}
= \operatorname{Retr}_{Θ}\!\left(
θ_k - η M(θ_k)^{-1} ∇_θ J(θ_k; θ⁻, y_{\mathrm{obs}})
\right),
\end{aligned}
```

where

- `M(θ)` is a chosen metric or preconditioner,
- `Retr_Θ` is a retraction or parametrization map that preserves valid Gaussian
  parameters, especially `Σ ≻ 0`.

If only one update step is desired, this metric-aware form is usually the
better interpretation of the KL-regularized objective. Taking `M(θ⁻)` to be the
local Hessian / Fisher metric induced by the Gaussian KL gives the first-order
approximation

```math
\begin{aligned}
θ⁺
\approx \operatorname{Retr}_{Θ}\!\left(
θ⁻ - η M(θ⁻)^{-1} ∇_θ[-\log q_ψ(y_{\mathrm{obs}} ∣ g_ψ(θ))]\big|_{θ=θ⁻}
\right),
\end{aligned}
```

which keeps the KL geometry in a single step.

Reasonable choices include:

- Euclidean coordinates on `(μ, L)` with `Σ = LLᵀ`,
- precision coordinates `(μ, Λ)` with `Λ = Σ⁻¹`,
- Cholesky or log-Cholesky coordinates for the covariance block,
- a Gaussian Fisher/KL metric for natural-gradient style updates.

Because `decoder.log_prob(y_obs, θ)` is explicit, the update never requires
Monte Carlo integration over `x`.

## History-Conditioned Path Forecasting

The per-time decoder `decode(θ_t)` gives the predictive marginal law at a single
time. To obtain a genuine path forecast, we need the joint law of the stacked
future latent or observation variables.

For query times `T = (t₁, …, tₙ)`, write

```math
\begin{aligned}
X_T &≔ (X_{t₁}, …, X_{tₙ}), \\
Y_T &≔ (Y_{t₁}, …, Y_{tₙ}).
\end{aligned}
```

The path-forecasting problem is to construct

```math
\begin{aligned}
p(Y_T ∈ \,\cdot \mid H),
\end{aligned}
```

not just the marginals `p(Y_{tᵢ} ∈ \,\cdot \mid H)`.

### How History Enters

Let the observation history be

```math
\begin{aligned}
H = \bigl((τ₁, y₁^{\mathrm{obs}}), …, (τ_m, y_m^{\mathrm{obs}})\bigr),
\qquad
τ₁ < \cdots < τ_m,
\end{aligned}
```

and let `(t₀, θ_{t₀}⁺)` be the initial anchored belief.

The history is assimilated by repeatedly applying the existing propagation and
update operators:

```math
\begin{aligned}
θ_{τ₁}⁻ &= \operatorname{propagate}(τ₁ - t₀, θ_{t₀}⁺), \\
θ_{τ₁}⁺ &= \operatorname{update}(y₁^{\mathrm{obs}}, θ_{τ₁}⁻, \operatorname{decoder}), \\
θ_{τ₂}⁻ &= \operatorname{propagate}(τ₂ - τ₁, θ_{τ₁}⁺), \\
θ_{τ₂}⁺ &= \operatorname{update}(y₂^{\mathrm{obs}}, θ_{τ₂}⁻, \operatorname{decoder}), \\
&\ \vdots \\
θ_{τ_m}⁻ &= \operatorname{propagate}(τ_m - τ_{m-1}, θ_{τ_{m-1}}⁺), \\
θ_{τ_m}⁺ &= \operatorname{update}(y_m^{\mathrm{obs}}, θ_{τ_m}⁻, \operatorname{decoder}).
\end{aligned}
```

Set

```math
\begin{aligned}
s &≔ τ_m, \\
θ_H &≔ θ_s⁺.
\end{aligned}
```

The statement

```text
"the history enters only through the filtered posterior"
```

is valid only in the **pure forecasting** regime, i.e. when all query times
satisfy `tᵢ ≥ s`.

It also needs to be read conditionally on any future exogenous information such
as covariates, controls, or known forcing terms. If `U` denotes the future
covariate path on `[s, \max T]`, then the correct statement is:

- for `tᵢ ≥ s`, the future latent path law depends on the history `H` only
  through the filtered posterior belief `p(· ∣ θ_H)`, **conditional on the
  future covariate path `U`**

That is,

```math
\begin{aligned}
p(X_T ∈ \,\cdot \mid H, U)
= \int p(X_T ∈ \,\cdot \mid x_s, U)\,p(x_s \mid θ_H)\,dx_s.
\end{aligned}
```

So for pure extrapolation, the history is summarized by `θ_H`, while the future
covariates must still be supplied explicitly.

If the future covariates are themselves uncertain rather than known, then one
must either:

- condition on a forecast/scenario for them, or
- model them jointly and integrate them out.

Equivalently,

```math
\begin{aligned}
p(X_T ∈ \,\cdot \mid H, U)
= \int p(X_T ∈ \,\cdot \mid x_s, U)\,p(x_s \mid θ_H)\,dx_s.
\end{aligned}
```

So the path forecaster does not need direct access to the full history once the
history has been summarized by the current posterior belief `θ_H`, provided we
are forecasting strictly after `s` and condition on the future covariate path.

### Path-Forecasting Interface

It is useful to make the history assimilation step explicit and then add two
path-level operators:

```text
filter_history : History × Time × Θ -> Time × Θ
propagate_path : Seq(Time) × Time × Θ × Covariates -> Θ_path
decode_path    : Θ_path -> 𝒫(Seq(𝒴))
```

with the intended meaning:

- `filter_history(H, t₀, θ₀⁺) -> (s, θ_H)` assimilates the history and returns
  the reference time `s` together with the filtered posterior `θ_H`.
- `propagate_path(T, s, θ_H, U) -> Θ_T` constructs the joint latent path law
  over the query times `T`, conditional on the future covariate path `U`.
- `decode_path(Θ_T)` maps that latent path law to a joint observation law.

The full path forecaster is then the composition

```text
path_forecast(T, H, U; t₀, θ₀⁺)
    = decode_path(propagate_path(T, s, θ_H, U))
      where (s, θ_H) = filter_history(H, t₀, θ₀⁺).
```

If the dynamics are time-homogeneous and the query is given in relative offsets
from the anchor time `s`, one may instead write

```text
propagate_path : Seq(Delta) × Θ × Covariates -> Θ_path.
```

However, for time-inhomogeneous linear ODE/SDE dynamics, the anchored form
`Seq(Time) × Time × Θ × Covariates -> Θ_path` is the cleaner signature.

### Exact Gaussian Path Law for Linear ODE/SDE Dynamics

Assume the latent dynamics are

```math
\begin{aligned}
dX_t = A_t X_t\,dt + a_t\,dt + G_t\,dW_t,
\end{aligned}
```

and that after assimilating the history we have the filtered posterior

```math
\begin{aligned}
X_s \mid H ∼ 𝓝(m_s⁺, P_s⁺).
\end{aligned}
```

Let `Φ(t, u)` denote the state-transition matrix from time `u` to time `t`, and
define

```math
\begin{aligned}
b(t, u) &≔ \int_u^t Φ(t, r)a_r\,dr, \\
Q(t, u) &≔ \int_u^t Φ(t, r)G_rG_rᵀΦ(t, r)ᵀ\,dr.
\end{aligned}
```

Then for query times `s ≤ t₁ < \cdots < tₙ`, the stacked latent path is exactly
Gaussian:

```math
\begin{aligned}
X_T \mid H ∼ 𝓝(m_T, K_T).
\end{aligned}
```

The block mean vector is

```math
\begin{aligned}
m_i = 𝐄[X_{tᵢ} \mid H] = Φ(tᵢ, s)m_s⁺ + b(tᵢ, s),
\end{aligned}
```

and the block covariance is

```math
\begin{aligned}
K_{ij}
= \operatorname{Cov}(X_{tᵢ}, X_{tⱼ} \mid H)
= Φ(tᵢ, s)P_s⁺Φ(tⱼ, s)ᵀ
+ \int_s^{\min(tᵢ, tⱼ)} Φ(tᵢ, r)G_rG_rᵀΦ(tⱼ, r)ᵀ\,dr.
\end{aligned}
```

So in the linear-Gaussian case,

```text
Θ_T = (m_T, K_T)
```

is a natural choice for the path-belief parameter.

This is the precise sense in which a linear ODE/SDE together with a Gaussian
filtered posterior induces a latent Gaussian process over future query times.

### Decoding the Latent Path

#### Kalman path decoder

For linear-Gaussian observations

```math
\begin{aligned}
Y_t = C_t X_t + c_t + V_t,
\qquad
V_t ∼ 𝓝(0, R_t),
\end{aligned}
```

the stacked observation path is again Gaussian:

```math
\begin{aligned}
Y_T \mid H ∼ 𝓝(m_T^Y, K_T^Y),
\end{aligned}
```

with block moments

```math
\begin{aligned}
m_i^Y &= C_{tᵢ} m_i + c_{tᵢ}, \\
K_{ij}^Y &= C_{tᵢ} K_{ij} C_{tⱼ}ᵀ + 𝟙_{i=j}R_{tᵢ}.
\end{aligned}
```

So the classical Kalman model yields an exact joint path forecast, not just
timewise marginals.

#### Neural path decoder

For the neural belief-space variant, a per-time decoder

```text
θ_t ↦ q(· ∣ g(θ_t))
```

only determines the marginal laws at each time. It does **not** determine the
joint law across time.

To obtain a genuine path forecast, the decoder itself must act on the full path
belief:

```text
g_path,ψ : Θ_path -> Ω_path
decode_path(Θ_path) = q_path(· ∣ g_path,ψ(Θ_path)).
```

For the Gaussian path belief `Θ_T = (m_T, K_T)`, tractable choices include:

- a joint Gaussian head on the stacked observation vector,
- a normalizing flow on the stacked observation vector, with parameters
  generated from `(m_T, K_T)`,
- an autoregressive decoder over `(Y_{t₁}, …, Y_{tₙ})` whose conditionals are
  explicit.

The important point is that the path decoder is conditioned on the Gaussian
path belief `Θ_T`, not on latent samples `x_t`. Therefore the latent path
construction remains exact and the decoder remains explicit.

### Filtering Versus Smoothing

The discussion above assumes the query times satisfy `tᵢ ≥ s`, i.e. we are
forecasting from the last assimilated observation time.

If instead the query times lie inside the observed window or before the final
history time, then one needs a **smoother** rather than a pure forecaster. In
that case the target is still

```math
\begin{aligned}
p(X_T ∈ \,\cdot \mid H)
\quad\text{or}\quad
p(Y_T ∈ \,\cdot \mid H),
\end{aligned}
```

but it is no longer obtained from the filtered posterior at the last history
time alone. In particular:

- if `T⁻ = T ∩ (-∞, s]` contains interpolation times, then `θ_H` alone is not a
  sufficient summary for `p(X_{T⁻} ∈ · \mid H)`
- one needs a smoothing object, not just the final filtered belief

A useful factorization for mixed interpolation/extrapolation is to split

```math
\begin{aligned}
T^- &≔ \{t ∈ T : t \le s\}, \\
T^+ &≔ \{t ∈ T : t > s\}.
\end{aligned}
```

Then the mixed-time target can be written as

```math
\begin{aligned}
p(X_{T^-}, X_{T^+} ∈ \,\cdot \mid H, U)
= \int p(X_{T^+} ∈ \,\cdot \mid x_s, U)\,
   p(X_{T^-}, x_s ∈ \,\cdot \mid H)\,dx_s.
\end{aligned}
```

So:

- the extrapolation part `T^+` is handled by forecasting forward from the
  anchor state at time `s`
- the interpolation part `T^-` requires the joint smoothing law with that anchor
  state

For linear-Gaussian models this is still exact and Gaussian via Kalman
smoothing. For nonlinear models it becomes an approximate smoothing problem.

## Summary Table

| Component | Interface signature | Kalman filter | EKF | Tractable Gaussian neural variant |
| --- | --- | --- | --- | --- |
| Propagation | `propagate(Δ, θ) -> θ_pred` | exact linear-Gaussian moment map | linearized nonlinear moment map | exact linear ODE/SDE Gaussian moment map |
| Decoder | `decode(θ) -> q(·)` | Gaussian `q(·)=𝓝(·; Hμ+c, HΣHᵀ+R)` | local Gaussian `q(·)=𝓝(·; h(μ), HΣHᵀ+R)` | neural `q_ψ(· ∣ g_ψ(θ))`, e.g. Gaussian or flow head |
| Update | `update(y_obs, θ_pred, decoder) -> θ_post` | exact Gaussian conditioning | conditioning on linearized measurement model | gradient/natural/proximal step on `λ KL - log q_ψ(y_obs ∣ g_ψ(θ))` |

## Recommended Interpretation

This interface cleanly separates three roles:

- **Propagation** is responsible only for temporal evolution of the latent
  belief.
- **Decoder** is responsible only for turning the latent belief into a
  predictive observation law.
- **Update** is responsible only for assimilating observations into the latent
  belief.

In this formulation, the Kalman filter is not a special-case API. It is simply
one concrete implementation where all three components are Gaussian and exact.
The EKF keeps the same signatures, but replaces exact Gaussian maps by local
Gaussian closures. The neural variant keeps the same signatures again, but uses
an expressive belief-space decoder together with a tractable optimization-based
update.
