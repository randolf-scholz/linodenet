# Classical State Space Model

$$
\begin{aligned}
   ̇x(t) &= f(t, x(t), u(t))  \\
   y(t) &= g(t, x(t), u(t))
\end{aligned}
$$

## Latent State Space Model (LSSM)

$$
\begin{aligned}
   ̇z(t) &= f(t, z(t))        \\
   z(t) &= ϕ(t, x(t), u(t))  \\
   y(t) &= g(t, x(t), u(t))
\end{aligned}
$$

We consider a split of the observables into 3 parts:

1. Autoregressive variables. These are variables that are unchanged by $g$.
2. Observables y(t). In practice, not all of them may be observed.
3. u(t): covariates. available for past, present and future.

## Linear Latent State Space Model (LiLSSM)

$$
\begin{aligned}
   ̇z(t) &= Az(t)             \\
   z(t) &= ϕ(t, x(t), u(t))  \\
   y(t) &= g(t, x(t), u(t))
\end{aligned}
$$

## Latent State Space Algorithm

```


for tᵢ, xᵢ, uᵢ in D:
    ∆tᵢ = tᵢ - tᵢ₋₁


```

## Variants

1. Concatenate $z = ϕ([x, u])$ and $ ̇z = Az$.
2. Embed individually: $z = ϕ(x)$, $v=ψ(u) and $ ̇z = Az + Bv$.
   - This is "more natural" in the sense that $u$ does not evolve as a dynamical system.
     Often, $u$ could be a step-function, so this assumption makes sense.
   - We call this variant the factorized LSSM (fLSSM).
   - On the other hand, $u$ could be such that it contains quasi-measurements.
     In this case, the model should learn to interpolate between the measurements.
   - Remark: We can solve the case when $u$ is piece-wise analytical without integrals.
