r"""Marchenko-Pastur distribution.

Let $X$ have shape $(m, n)$ with independent, identically distributed entries.
As $n → ∞$ with $m/n → γ > 0$:

- For entries $𝓝(0, σ²/n)$, the empirical eigenvalue distribution of $XXᵀ$
    converges to $\operatorname{MP}(γ, σ²)$.
- For entries $𝓝(0, σ²)$, the empirical eigenvalue distribution of $XXᵀ/n$
    has the same limit. The unnormalized spectrum is scaled by $n$, corresponding
    to the asymptotic approximation $\operatorname{MP}(γ, σ²n)$.

Zero eigenvalues are included when $m > n$.

The continuous part of the law with parameters $γ$ and $σ²$ has support

$$  [λ₋, λ₊] = [σ²(1 - √γ)², σ²(1 + √γ)²]   $$

and density, for $x > 0$ in this interval,

$$  f(x) = \frac{\sqrt{(λ₊ - x)(x - λ₋)}}{2πσ²γx}.  $$

For $γ > 1$, the law also has an atom at $x = 0$ with mass $1 - 1/γ$.
"""

__all__ = ["MarchenkoPastur", "Union"]

import math

import torch
from torch import Generator, Tensor
from torch.distributions import constraints

from .base import DistributionBase

type Size = tuple[int, ...] | list[int]


class Union(constraints.Constraint):
    r"""Constraint satisfied when any constituent constraint is satisfied."""

    def __init__(self, *constraints_: constraints.Constraint) -> None:
        super().__init__()
        self.constraints = constraints_

    def check(self, value: Tensor) -> Tensor:
        if not self.constraints:
            return torch.zeros_like(value, dtype=torch.bool)
        checks: list[Tensor] = [
            constraint.check(value) for constraint in self.constraints
        ]
        return torch.stack(checks).any(dim=0)


class MarchenkoPastur(DistributionBase):
    r"""Marchenko-Pastur distribution with positive parameters $γ$ and $σ²$."""

    arg_constraints = {
        "gamma": constraints.positive,
        "sigma2": constraints.positive,
    }
    has_rsample = False

    def __init__(
        self,
        gamma: Tensor | float,
        sigma2: Tensor | float = 1.0,
        *,
        validate_args: bool | None = None,
    ) -> None:
        gamma_t = torch.as_tensor(gamma)
        sigma2_t = torch.as_tensor(sigma2)
        batch_shape = torch.broadcast_shapes(gamma_t.shape, sigma2_t.shape)
        self.gamma = gamma_t.expand(batch_shape)
        self.sigma2 = sigma2_t.expand(batch_shape)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=torch.Size(),
        )

        if validate_args:
            for key, constraint in self.arg_constraints.items():
                param = getattr(self, key)
                if not constraint.check(param).all():
                    raise ValueError(
                        f"Invalid parameter {key}={param} for {self.__class__.__name__}. "
                        f"Must satisfy constraint {constraint}."
                    )

    @property
    def support(self) -> constraints.Constraint:
        interval = constraints.interval(self.lower_bound, self.upper_bound)
        if torch.any(self.gamma > 1):
            zero = torch.zeros_like(self.lower_bound)
            atom = constraints.interval(zero, zero)
            return Union(atom, interval)
        return interval

    @property
    def lower_bound(self) -> Tensor:
        return self.sigma2 * (1 - torch.sqrt(self.gamma)) ** 2

    @property
    def upper_bound(self) -> Tensor:
        return self.sigma2 * (1 + torch.sqrt(self.gamma)) ** 2

    @property
    def mean(self) -> Tensor:
        return self.sigma2

    @property
    def variance(self) -> Tensor:
        return self.sigma2.pow(2) * self.gamma

    @property
    def skewness(self) -> Tensor:
        return torch.sqrt(self.gamma)

    @property
    def point_mass(self) -> Tensor:
        return torch.where(self.gamma > 1, 1 - (1 / self.gamma), 0.0)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Compute the log probability density function of the Marchenko-Pastur distribution.

        For $x > 0$ in $[λ₋, λ₊]$, where $λ₋ = σ²(1 - √γ)²$ and
        $λ₊ = σ²(1 + √γ)²$:

        $$  \log p(x) = ½ \log((λ₊ - x)(x - λ₋)) - \log(2πσ²γx).    $$

        At $x = 0$, return the log mass of the atom, with weight
        $\max(0, 1 - 1/γ)$. Values outside the support have log probability $-∞$.
        """
        x = value
        a = self.lower_bound
        b = self.upper_bound
        in_support = (x >= a) & (x <= b)
        term = (b - x) * (x - a)

        value = 0.5 * torch.log(term) - torch.log(
            2 * math.pi * self.sigma2 * self.gamma * x
        )

        return torch.where(
            x == 0,
            torch.log(self.point_mass),
            torch.where(in_support, value, -math.inf),
        )

    def cdf(self, value: Tensor) -> Tensor:
        x = value
        a = self.lower_bound
        b = self.upper_bound
        c = (a + b) / 2
        m = torch.sqrt(a * b)
        db = b - x
        da = x - a

        value = (
            torch.sqrt(db * da)
            - c * torch.arccos(-(db - da) / (b - a))
            + 2 * m * torch.arctan(torch.sqrt((a * db) / (b * da)))
            + math.pi * (c - m)
        ) / (2 * math.pi * self.sigma2 * self.gamma)

        jump = torch.where(x >= 0, self.point_mass, 0.0)
        return torch.where(
            x <= a,
            jump,
            torch.where(x >= b, 1.0, value + jump),
        )

    def icdf(self, value: Tensor) -> Tensor:
        point_mass = torch.broadcast_to(self.point_mass, value.shape)
        target = torch.clamp(value, min=0.0, max=1.0)
        inv = _icdf_bisect(self, target)
        return torch.where(target <= point_mass, 0.0, inv)

    def sample(
        self, sample_shape: int | Size = (), rng: Generator | None = None
    ) -> Tensor:
        sample_shape = (
            (sample_shape,) if isinstance(sample_shape, int) else tuple(sample_shape)
        )
        shape = tuple(sample_shape) + self.batch_shape
        value = torch.rand(
            shape,
            device=self.gamma.device,
            dtype=self.gamma.dtype,
            generator=rng,
        )
        return self.icdf(value)

    def sample_positive(
        self,
        sample_shape: Size = (),
        rng: Generator | None = None,
    ) -> Tensor:
        r"""Sample from the Marchenko-Pastur distribution conditioned on $x > 0$.

        For $γ > 1$, the Marchenko-Pastur law has an atom at $x = 0$ with mass
        $1 - 1/γ$. This method excludes that atom and samples from the
        renormalized continuous part on $[λ₋, λ₊]$. For $γ ≤ 1$, this is identical
        to [`sample`][imtskit.distributions.marchenko_pastur.MarchenkoPastur.sample].
        """
        shape = tuple(sample_shape) + self.batch_shape
        value = torch.rand(
            shape,
            device=self.gamma.device,
            dtype=self.gamma.dtype,
            generator=rng,
        )
        point_mass = torch.broadcast_to(self.point_mass, shape)
        return self.icdf(point_mass + (1 - point_mass) * value)


def _icdf_bisect(
    dist: MarchenkoPastur,
    value: Tensor,
    /,
    *,
    maxiter: int = 128,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tensor:
    r"""Invert the CDF with bisection for values in (0, 1)."""
    shape = torch.broadcast_shapes(value.shape, dist.batch_shape)
    lo = torch.broadcast_to(dist.lower_bound, shape).clone()
    hi = torch.broadcast_to(dist.upper_bound, shape).clone()

    for _ in range(maxiter):
        mid = (lo + hi) * 0.5
        cdf_mid = dist.cdf(mid)
        lo = torch.where(cdf_mid < value, mid, lo)
        hi = torch.where(cdf_mid >= value, mid, hi)
        width = (hi - lo).abs()
        tol = atol + rtol * mid.abs()
        if torch.max(width - tol).item() <= 0:
            break

    return (lo + hi) * 0.5
