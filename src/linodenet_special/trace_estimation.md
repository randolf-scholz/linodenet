# Trace Estimation with Hutchinson's Method and XTrace

## Hutchinson's Method

$$ \tr(A) = 𝐄[vᴴAv] \qquad\text{where}\qquad 𝐄[vvᴴ] = 𝕀 $$

More generally, this also works with bilinear forms:

$$ \tr(A) = 𝐄[uᴴAv] \qquad\text{where}\qquad 𝐄[uvᴴ] = 𝕀 $$

Proof:

$$ 𝐄[uᴴAv] = 𝐄[ ⟨1∣uᴴAv⟩ ] = 𝐄[ ⟨uvᴴ∣A⟩ ] = ⟨𝐄[uvᴴ]∣A⟩ = ⟨𝕀∣A⟩ = \tr(A) $$

In practice, we choose a finite number of probe vectors $v₁, v₂, \dots, vₘ$ and compute the average.

## Hutch++

Hutch++ is a variance reduction technique that combines Hutchinson's method with a low-rank approximation of $A$.
It works in two steps:

1. compute the trace of a low-rank approximation of $A$ using a small number of probe vectors, and
2. compute the trace of the residual using Hutchinson's method, using projected probe vectors.

which is based on the formula

$$ \tr(A) = \tr(QQ^*AQQ^*) + \tr((I-QQ^*)A(I-QQ^*)) $$

Algorithm:

$$ \begin{aligned}
    V &= [v₁, ..., vₘ] ∈ ℝ^{n×m}
\\  Q, R &= \orth(AV)   \qquad\text{economical QR, so $Q$ has $m$ columns}
\\  G &= (𝕀 - QQᴴ)ᴴA(𝕀 - QQᴴ)
\\  \tr &≈ \tr(QᴴAQ) + hutchinson(G, m) \qquad\text{using independent samples}
\end{aligned} $$

Here $QᴴAQ$ is the trace of the $m×m$ low-rank approximation,
and the second term is the Hutchinson estimate of the residual,
after projecting the probe vectors to be orthogonal to the low-rank subspace.
($I-QQᴴ$ is the projection matrix onto the orthogonal complement of the column space of $Q$.)

## XTrace

XTrace is a refinement of the Hutch++ method, which reduces variance through orthogonalization.
XTrace satisfies an important exchangeability property:

> If the probe vectors are exchangeable, that is the joint distribution of the probe vectors is invariant under permutations,
then the trace estimates are invariant under permutations of the probe vectors.

$$\begin{aligned}
    V &= [v₁, ..., vₘ] ∈ ℝ^{n×m}
\\  Qᵢ &= \orth(AV₋₁)   \qquad\text{economical QR, so $Qᵢ$ is n×(m-1)}
\\  \trᵢ &= \tr(QᵢᴴAQᵢ) + vᵢᴴ(𝕀 - QᵢQᵢᴴ)ᴴA(I-QᵢQᵢᴴ)vᵢ
\\  \tr &≈ \frac{1}{m} \sum_{i=1}^m \trᵢ
\end{aligned}$$

The main trick is to use a rank-1 update to avoid computing $Qᵢ$ from scratch for each $i$:

$$\begin{aligned}
    QᵢQᵢᴴ &= Q(𝕀-sᵢsᵢᴴ)Qᴴ
    \qquad\text{where}\qquad
    R₋ᵢᴴsᵢ = 0, ‖sᵢ‖=1, AV = QR
\\  ⇝ \tr(QᵢᴴAQᵢ) &= \tr(AQᵢQᵢᴴ) = \tr(AQ(𝕀-sᵢsᵢᴴ)Qᴴ)
\\      &= \tr(QᴴAQ(𝕀-sᵢsᵢᴴ)) = \tr(QᴴAQ) - sᵢᴴQᴴAQsᵢ
\\  ⇝ vᵢᴴ(𝕀 - QᵢQᵢᴴ)A(I-QᵢQᵢᴴ)vᵢ
\\        &= vᵢᴴ(𝕀 - Q(𝕀-sᵢsᵢᴴ)Qᴴ)ᴴA(I-Q(𝕀-sᵢsᵢᴴ)Qᴴ)vᵢ
\\        &= vᵢᴴ(𝕀 - QQᴴ + QsᵢsᵢᴴQᴴ)A(𝕀 - QQᴴ + QsᵢsᵢᴴQᴴ)vᵢ
\end{aligned}$$

expanding the last line, we get 9 terms:

$$\begin{array}{lll}
    +vᵢᴴAvᵢ       & -vᵢᴴAQQᴴvᵢ            & +vᵢᴴAQsᵢsᵢᴴQᴴvᵢ \\
  -vᵢᴴQQᴴAvᵢ      & +vᵢᴴQQᴴAQQᴴvᵢ         & -vᵢᴴQQᴴAQsᵢsᵢᴴQᴴvᵢ \\
  +vᵢᴴQsᵢsᵢᴴQᴴAvᵢ & -vᵢᴴQsᵢsᵢᴴQᴴAQQᴴvᵢ    & +vᵢᴴQsᵢsᵢᴴQᴴAQsᵢsᵢᴴQᴴvᵢ
\end{array}$$

Since $AV=QR$, we have $(I-QQᴴ)AV=0$:

$$\begin{array}{lll}
        0         & -vᵢᴴAQQᴴvᵢ            & +vᵢᴴAQsᵢsᵢᴴQᴴvᵢ \\
        0         & +vᵢᴴQQᴴAQQᴴvᵢ         & -vᵢᴴQQᴴAQsᵢsᵢᴴQᴴvᵢ \\
  +vᵢᴴQsᵢsᵢᴴQᴴAvᵢ & -vᵢᴴQsᵢsᵢᴴQᴴAQQᴴvᵢ    & +vᵢᴴQsᵢsᵢᴴQᴴAQsᵢsᵢᴴQᴴvᵢ
\end{array}$$

Next, substitute $H = QᴴAQ$, and $W = QᴴV$, $QᴴAV=R$, and $tᵢ=vᵢᴴAQ$ we get:

$$\begin{array}{lll}
                    & -⟨tᵢ∣wᵢ⟩        & +⟨tᵢ∣sᵢ⟩⟨sᵢ∣wᵢ⟩
\\  0               & +wᵢᴴHwᵢ         & -wᵢᴴHsᵢ ⟨sᵢ∣wᵢ⟩
\\  +⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ⟩ & -⟨wᵢ∣sᵢ⟩sᵢᴴHwᵢ  & +⟨wᵢ∣sᵢ⟩sᵢᴴHsᵢ⟨sᵢ∣wᵢ⟩
\end{array}$$

Next, set $xᵢ = (I-sᵢsᵢᴴ)wᵢ = wᵢ - ⟨sᵢ∣wᵢ⟩sᵢ$, then:

$$ \begin{aligned}
    xᵢᴴHxᵢ &= wᵢᴴHwᵢ - wᵢᴴHsᵢ⟨sᵢ∣wᵢ⟩ - ⟨wᵢ∣sᵢ⟩sᵢᴴHwᵢ + ⟨wᵢ∣sᵢ⟩sᵢᴴHsᵢ⟨sᵢ∣wᵢ⟩
\\ ⟨tᵢ∣xᵢ⟩ &= ⟨tᵢ∣wᵢ⟩ - ⟨tᵢ∣sᵢ⟩⟨sᵢ∣wᵢ⟩
\end{aligned}$$

So, the terms combine to, which is formula SM3.3 in the paper:

$$ xᵢᴴHxᵢ - ⟨tᵢ∣xᵢ⟩ + ⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ⟩ $$

Alternatively, we can aggregate the terms differently:

$$\begin{array}{lll}
    0               & -⟨tᵢ∣wᵢ⟩        & +⟨tᵢ∣sᵢ⟩⟨sᵢ∣wᵢ⟩
\\  0               & +wᵢᴴHwᵢ         & -wᵢᴴHsᵢ ⟨sᵢ∣wᵢ⟩
\\  +⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ⟩ & -⟨wᵢ∣sᵢ⟩sᵢᴴHwᵢ  & +⟨wᵢ∣sᵢ⟩sᵢᴴHsᵢ⟨sᵢ∣wᵢ⟩
\end{array}$$

combining (1,3) and (2,3) we get: $⟨sᵢ∣wᵢ⟩⟨tᵢ - Hᴴwᵢ∣sᵢ⟩$
combining (3,1) and (3,2) we get: $⟨wᵢ∣sᵢ⟩⟨sᵢ∣rᵢ - Hwᵢ⟩$
combining (1,2) and (2,2) we get: $-⟨tᵢ - Hᴴwᵢ∣wᵢ⟩$
simplifying (3,3) we get: $|⟨sᵢ∣wᵢ⟩|²⟨sᵢ∣Hsᵢ⟩$

which gives in total the following, which is how the paper actually implements the XTrace estimator:

$$\begin{aligned}
|⟨sᵢ∣wᵢ⟩|²⟨sᵢ∣Hsᵢ⟩ + \overline{⟨sᵢ∣wᵢ⟩}⟨sᵢ∣rᵢ - Hwᵢ⟩ - ⟨tᵢ - Hᴴwᵢ∣wᵢ - ⟨sᵢ∣wᵢ⟩sᵢ⟩
\end{aligned}$$

## BiHutch++

We extend Hutch++ to the bilinear case, which we call BHutch++.
The basic idea is again to use a low-rank approximation: $\tr(PᴴAQ)$

$$\begin{aligned}
    U  &= [u₁, ..., uₘ] ∈ ℝ^{n×m}
\\  V  &= [v₁, ..., vₘ] ∈ ℝ^{n×m}
\\  P &= \orth(AᴴU)   \qquad\text{economical QR, so $P$ has $m$ columns}
\\  Q &= \orth(AV)    \qquad\text{economical QR, so $Q$ has $m$ columns}
\\  \tr &≈ \tr(PᴴAQ) + \frac{1}{m} \sum_{i=1}^m uᵢᴴ(𝕀 - PPᴴ)A(𝕀 - QQᴴ)vᵢ
\end{aligned}$$

### Bi-Orthogonal Variant

Instead of separately orthogonalizing $U$ and $V$, we can use bi-orthogonalization:

$$\begin{aligned}
    U &= [u₁, ..., uₘ] ∈ ℝ^{n×m}
\\  V &= [v₁, ..., vₘ] ∈ ℝ^{n×m}
\\  EΣFᴴ &= SVD(UᴴAV)   \qquad\text{so $E$ has $m$ columns, and $F$ has $m$ columns}
\\  P &= UEΣ⁻¹ᐟ²   \qquad\text{so $P$ has $m$ columns}
\\  Q &= VFΣ⁻¹ᐟ²   \qquad\text{so $Q$ has $m$ columns}
\\  \tr &≈ \tr(PᴴAQ) + \frac{1}{m} \sum_{i=1}^m uᵢᴴ(𝕀 - PPᴴ)A(𝕀 - QQᴴ)vᵢ
\end{aligned}$$

⟨B∣I⟩
