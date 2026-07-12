r"""Test Gaussian distribution utilities."""

import pytest
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from linodenet.distributions.gaussian import (
    CovarianceType,
    argmin_forward_kl,
    argmin_proximal_reverse_kl,
    argmin_reverse_kl,
    fisher,
    inverse_fisher,
    kl,
    log_prob,
    solve_proximal_reverse_kl,
)
from tests.testing import SEEDS_3

BATCH_SHAPES = [(), (6,), (1, 2, 3)]
GAMMA_MODES = ["scalar", "batched", "split-batched"]
RHO_MODES = ["scalar", "batched", "split-batched"]


@pytest.fixture(params=SEEDS_3, ids="seed={}".format)
def seed(request: pytest.FixtureRequest) -> int:
    r"""Return a reproducible seed for randomized Gaussian distribution tests."""
    return int(request.param)


def _symmetric(matrix: Tensor, /) -> Tensor:
    r"""Return the symmetric part of a square matrix."""
    return 0.5 * (matrix + matrix.mT)


def _directional_second_derivative(fn, /) -> Tensor:
    r"""Return the second derivative of `fn(t)` at `t = 0`."""
    t = torch.zeros((), requires_grad=True)
    value = fn(t)
    gradient = torch.autograd.grad(value, t, create_graph=True)[0]
    return torch.autograd.grad(gradient, t)[0]


def _parameter_inner_product(
    left: tuple[Tensor, Tensor],
    right: tuple[Tensor, Tensor],
    /,
) -> Tensor:
    r"""Return the Euclidean/Frobenius inner product on Gaussian parameters."""
    left_mean, left_matrix = left
    right_mean, right_matrix = right
    return (left_mean * right_mean).sum(dim=-1) + (left_matrix * right_matrix).sum(
        dim=(-2, -1)
    )


def _make_gamma(
    batch_shape: tuple[int, ...], mode: str, /
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Return a shared or split test gamma with optional batch shape."""
    match mode:
        case "scalar":
            return torch.tensor(1.7)
        case "batched":
            return torch.full(batch_shape, 1.7) if batch_shape else torch.tensor(1.7)
        case "split-batched":
            gamma_mu = (
                torch.full(batch_shape, 1.3) if batch_shape else torch.tensor(1.3)
            )
            gamma_sigma = (
                torch.full(batch_shape, 2.1) if batch_shape else torch.tensor(2.1)
            )
            return gamma_mu, gamma_sigma
        case other:
            raise AssertionError(f"Unexpected gamma mode: {other!r}")


def _make_rho(
    batch_shape: tuple[int, ...], mode: str, /
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Return a shared or split test rho with optional batch shape."""
    match mode:
        case "scalar":
            return torch.tensor(1.7 / 2.7)
        case "batched":
            rho = 1.7 / 2.7
            return torch.full(batch_shape, rho) if batch_shape else torch.tensor(rho)
        case "split-batched":
            rho_mu = (
                torch.full(batch_shape, 1.3 / 2.3)
                if batch_shape
                else torch.tensor(1.3 / 2.3)
            )
            rho_sigma = (
                torch.full(batch_shape, 2.1 / 3.1)
                if batch_shape
                else torch.tensor(2.1 / 3.1)
            )
            return rho_mu, rho_sigma
        case other:
            raise AssertionError(f"Unexpected rho mode: {other!r}")


def _make_reverse_rho(
    batch_shape: tuple[int, ...], mode: str, /
) -> tuple[Tensor, Tensor]:
    r"""Return reverse-KL retentions with optional batch shape."""
    match mode:
        case "scalar":
            gamma = 1.7
            return torch.tensor(gamma / (1.0 + gamma)), torch.tensor(
                (gamma - 1.0) / gamma
            )
        case "batched":
            gamma = 1.7
            rho_mu = gamma / (1.0 + gamma)
            rho_sigma = (gamma - 1.0) / gamma
            if batch_shape:
                return torch.full(batch_shape, rho_mu), torch.full(
                    batch_shape, rho_sigma
                )
            return torch.tensor(rho_mu), torch.tensor(rho_sigma)
        case "split-batched":
            gamma_mu = 1.3
            gamma_sigma = 2.1
            rho_mu = gamma_mu / (1.0 + gamma_mu)
            rho_sigma = (gamma_sigma - 1.0) / gamma_sigma
            if batch_shape:
                return torch.full(batch_shape, rho_mu), torch.full(
                    batch_shape, rho_sigma
                )
            return torch.tensor(rho_mu), torch.tensor(rho_sigma)
        case other:
            raise AssertionError(f"Unexpected reverse rho mode: {other!r}")


def _solve_reverse_kl_bisection(q: Tensor, gamma: Tensor, /) -> Tensor:
    r"""Solve the forward-KL scalar stationarity equation by bisection."""
    if gamma.ndim != 0:
        raise ValueError("Expected gamma to be a float or scalar tensor.")

    beta = (gamma - 1) / gamma

    if torch.any(gamma <= 1).item():
        raise ValueError(
            "The exact forward-KL Gaussian update requires gamma > 1. "
            "For gamma <= 1 the objective does not admit a finite minimizer."
        )

    if not torch.any(q > 0).item():
        return beta

    def residual(variance: Tensor, /) -> Tensor:
        return (
            gamma
            + (1 - gamma) / variance
            - gamma.square() * q / (1 + gamma * variance).square()
        )

    lower = beta
    upper = beta + torch.ones_like(beta)
    need_expand = (q > 0) & (residual(upper) < 0)

    while torch.any(need_expand).item():
        upper = torch.where(need_expand, 2 * upper, upper)
        need_expand = (q > 0) & (residual(upper) < 0)

    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        go_left = residual(midpoint) >= 0
        upper = torch.where((q > 0) & go_left, midpoint, upper)
        lower = torch.where((q > 0) & ~go_left, midpoint, lower)

    return torch.where(q > 0, upper, beta)


class TestReverseKLSolvers:
    r"""Tests for the scalar forward-KL cubic solvers."""

    @staticmethod
    def _solve_s_closed_form(
        sq_dist: Tensor,  # (...)
        gamma_mean: Tensor,  # (...)
        gamma_cov: Tensor,  # (...)
        /,
        *,
        use_fp64: bool = True,
    ) -> Tensor:  # (...)
        r"""Solve the reverse-KL scalar stationarity equation for the positive branch.

        Returns the unique admissible root $s>0$ of

            (1 − γ_Σ)/s + γ_Σ − γ_μ²·q / (1 + γ_μ·s)² = 0,   γ_μ ≥ 0,  γ_Σ > 1,  q ≥ 0.

        For $γ_μ>0$, substituting $u = 1 + γ_μ·s$ gives the monic cubic $u³ + a·u² + b·u − b$,
        with $β = (γ_Σ − 1)/γ_Σ$, $a = −(1 + γ_μ·β)$, $b = −γ_μ²·q/γ_Σ$. Its roots are one
        negative, one in $(0,1)$ and one $>1$ (the product of the roots is $b ≤ 0$ and
        $f(0) ≥ 0 > f(1)$); only $u>1$ gives $s>0$, so the admissible root is the largest and
        $s = (u − 1)/γ_μ$. Depressing with $u = t − a/3$ always yields $p ≤ −1/3 < 0$, so this
        is the casus irreducibilis and the largest root is the $k=0$ cosine branch.

        Note: Degeneracy parameter
            Both special cases are governed by $B ≔ -b = γ_μ²·q/γ_Σ$. Writing the stationarity
            condition as the fixed point $s = β + s·B/(1 + γ_μ·s)²$ and expanding once gives

                s = β·(1 + B/(1 + γ_μ·β)²) + O(B²),

            which is exact at $B = 0$ (i.e. at $γ_μ = 0$, where the mean snaps to the
            observation, *and* at $q = 0$, where the observation is already at the prior mean).
            The $(1 + γ_μ·β)^{-2}$ factor is what makes this $O(B²)$ rather than $O(B·γ_μ)$;
            at fixed $q$ that is $O(γ_μ⁴)$.

        Note: Why two thresholds
            `B < eps**0.5` guards the cosine endpoint (below). `γ_μ < eps**0.25` guards a
            different failure: as $γ_μ → 0$ we have $u → 1$, so the exact path evaluates
            $(u − 1)/γ_μ$ as a ratio of two vanishing quantities and silently loses
            $≈ log₁₀(1/γ_μβ)$ digits well before $γ_μ$ underflows. The two overlap in
            practice but are not nested: for $γ_μ ≳ 900$ the cosine can reach its endpoint
            while $B$ is still above the first threshold.

        Note: NaN-safe branching
            `torch.where` backpropagates through *both* arms, so every discarded arm must be
            finite in value *and* in local derivative — a $0 · ∞$ in the dead branch poisons
            the gradient of the live one.

            - `γ_μ_safe` is threaded through `a` and `b`, not just the final division, so the
              dead cubic never divides by zero.
            - $\cos θ → 1⁻$ as $B → 0$ (specifically $1 - \cosθ ≈ 9B/2(1+γ_μβ)³$) and rounds
              to $≥ 1$ there, where $\arccos'(1) = -∞$. Clamping alone fixes the forward value
              but leaves $0 · (−∞) = \mathrm{NaN}$ in $∂s/∂q$, so those entries are additionally
              folded into `degenerate` and their cosine argument is neutralized to $0$.

            By contrast $-p ≥ 1/3$ and $m³ ≥ 1/27$ are bounded away from zero and need no guard.
        """
        q, γ_μ, γ_Σ = torch.broadcast_tensors(sq_dist, gamma_mean, gamma_cov)
        out_dtype = torch.promote_types(
            q.dtype, torch.promote_types(γ_μ.dtype, γ_Σ.dtype)
        )
        work_dtype = (
            torch.float64 if use_fp64 else torch.promote_types(out_dtype, torch.float32)
        )

        γ_μ = γ_μ.to(work_dtype)
        γ_Σ = γ_Σ.to(work_dtype)
        q = q.to(work_dtype)
        eps = torch.finfo(work_dtype).eps
        β = (γ_Σ - 1.0) / γ_Σ

        # Series branch in the degeneracy parameter B = -b; exact at B = 0, error O(B²).
        B = γ_μ.square() * q / γ_Σ
        s_series = β * (1.0 + B / (1.0 + γ_μ * β).square())
        small = (B < eps**0.5) | (γ_μ < eps**0.25)

        γ_μ_safe = torch.where(small, 1.0, γ_μ)
        a = -(1.0 + γ_μ_safe * β)
        b = -γ_μ_safe.square() * q / γ_Σ
        c = -b
        p = b - a.square() / 3.0
        r = 2.0 * a.pow(3) / 27.0 - (a * b) / 3.0 + c

        # In the admissible regime p < 0, so the largest real root uses the cosine form.
        m = torch.sqrt(-p / 3.0)
        raw = -r / (2.0 * m.pow(3))

        # acos'(±1) = ∓∞ and `where` backprops through the dead arm, so neutralize the
        # argument wherever the cosine sits on its endpoint and take the series instead.
        degenerate = small | (raw >= 1.0)
        cos_θ = torch.where(degenerate, torch.zeros_like(raw), raw).clamp(-1.0, 1.0)

        t = 2.0 * m * torch.cos(torch.acos(cos_θ) / 3.0)
        u = t - a / 3.0
        s_exact = (u - 1.0) / γ_μ_safe

        return torch.where(degenerate, s_series, s_exact).to(dtype=out_dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_closed_form_matches_bisection(self, dtype: torch.dtype) -> None:
        r"""Test that the closed-form cubic solver matches the bisection solver."""
        gamma_grid = torch.tensor(
            [1.0001, 1.001, 1.01, 1.1, 1.7, 2.0, 10.0], dtype=dtype
        )
        q_grid = torch.tensor(
            [0.0, 1e-12, 1e-8, 1e-4, 1e-2, 1.0, 10.0, 1e3, 1e6],
            dtype=dtype,
        )
        gamma_random = 1 + torch.rand(64, dtype=dtype) * 1e3
        q_random = torch.rand(64, dtype=dtype) * 1e6
        gamma = torch.cat([gamma_grid, gamma_random])
        q = torch.cat([q_grid, q_random])

        atol = 2e-4 if dtype is torch.float32 else 1e-10
        rtol = 1e-6 if dtype is torch.float32 else 1e-12

        for gamma_value in gamma:
            expected = _solve_reverse_kl_bisection(q, gamma_value)
            actual = self._solve_s_closed_form(q, gamma_value, gamma_value)
            assert torch.allclose(actual, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
    def test_closed_form_solves_stationarity_equation_with_finite_gradients(
        self, dtype: torch.dtype
    ) -> None:
        r"""Test that the closed-form solver satisfies its stationarity equation."""
        gamma_mean_values = torch.tensor(
            [0.0, 1e-8, 1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3],
            dtype=dtype,
        )
        gamma_cov_values = torch.tensor(
            [1.0 + 1e-4, 1.001, 1.1, 2.0, 10.0],
            dtype=dtype,
        )
        q_values = torch.tensor(
            [0.0, 1e-12, 1e-8, 1e-4, 1e-2, 1.0, 10.0, 1e3, 1e6],
            dtype=dtype,
        )

        q, gamma_mean, gamma_cov = torch.meshgrid(
            q_values, gamma_mean_values, gamma_cov_values, indexing="ij"
        )
        q = q.clone().requires_grad_()
        gamma_mean = gamma_mean.clone().requires_grad_()
        gamma_cov = gamma_cov.clone().requires_grad_()

        actual = self._solve_s_closed_form(q, gamma_mean, gamma_cov)
        residual = (
            (1.0 - gamma_cov) / actual
            + gamma_cov
            - gamma_mean.square() * q / (1.0 + gamma_mean * actual).square()
        )
        scale = (
            ((1.0 - gamma_cov) / actual).abs()
            + gamma_cov.abs()
            + (gamma_mean.square() * q / (1.0 + gamma_mean * actual).square()).abs()
        )

        atol = 2e-4 if dtype is torch.float32 else 2e-8
        rtol = 2e-5 if dtype is torch.float32 else 1e-8

        assert actual.shape == q.shape
        assert torch.all(actual > 0)
        assert torch.all(residual.abs() <= atol + rtol * scale)

        grads = torch.autograd.grad(actual.sum(), (q, gamma_mean, gamma_cov))
        for grad in grads:
            assert grad is not None
            assert torch.isfinite(grad).all()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
@pytest.mark.parametrize("parametrization", CovarianceType)
def test_kl_matches_torch_distribution(
    parametrization: CovarianceType,
    batch_shape: tuple[int, ...],
) -> None:
    r"""Test the closed-form KL divergence against PyTorch across batch shapes."""
    dim = 4

    mean_p = torch.randn(*batch_shape, dim)
    mean_q = torch.randn(*batch_shape, dim)

    factor_p = torch.randn(*batch_shape, dim, dim)
    factor_q = torch.randn(*batch_shape, dim, dim)
    eye = torch.eye(dim)
    cov_p = factor_p @ factor_p.mT + eye
    cov_q = factor_q @ factor_q.mT + eye
    chol_p = torch.linalg.cholesky(cov_p)
    chol_q = torch.linalg.cholesky(cov_q)
    log_chol_p = chol_p.tril(diagonal=-1) + torch.diag_embed(
        chol_p.diagonal(dim1=-2, dim2=-1).log()
    )
    log_chol_q = chol_q.tril(diagonal=-1) + torch.diag_embed(
        chol_q.diagonal(dim1=-2, dim2=-1).log()
    )

    theta_p = {
        "covariance": (mean_p, cov_p),
        "precision": (mean_p, torch.linalg.inv(cov_p)),
        "cholesky": (mean_p, chol_p),
        "log-cholesky": (mean_p, log_chol_p),
    }[parametrization]
    theta_q = {
        "covariance": (mean_q, cov_q),
        "precision": (mean_q, torch.linalg.inv(cov_q)),
        "cholesky": (mean_q, chol_q),
        "log-cholesky": (mean_q, log_chol_q),
    }[parametrization]
    actual = kl(theta_p, theta_q, parametrization=parametrization)
    expected = kl_divergence(
        MultivariateNormal(mean_p, covariance_matrix=cov_p),
        MultivariateNormal(mean_q, covariance_matrix=cov_q),
    )

    assert actual.shape == batch_shape
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    "sample_shape", [(), (5,), (4, 2)], ids="sample_shape={}".format
)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
@pytest.mark.parametrize("parametrization", CovarianceType)
def test_log_prob_matches_torch_distribution(
    parametrization: CovarianceType,
    sample_shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> None:
    r"""Test the Gaussian log-density against PyTorch in all parametrizations."""
    dim = 4

    mean = torch.randn(*batch_shape, dim)
    factor = torch.randn(*batch_shape, dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)
    precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    chol = torch.linalg.cholesky(covariance)
    log_chol = chol.tril(diagonal=-1) + torch.diag_embed(
        chol.diagonal(dim1=-2, dim2=-1).log()
    )
    value = torch.randn(*sample_shape, *batch_shape, dim)

    theta = {
        "covariance": (mean, covariance),
        "precision": (mean, precision),
        "cholesky": (mean, chol),
        "log-cholesky": (mean, log_chol),
    }[parametrization]
    actual = log_prob(value, theta, parametrization=parametrization)
    expected = MultivariateNormal(mean, covariance_matrix=covariance).log_prob(value)

    assert actual.shape == (*sample_shape, *batch_shape)
    assert torch.allclose(actual, expected)


def test_log_prob_rejects_unknown_parametrization() -> None:
    r"""Test that the public log-density dispatch rejects unknown parametrizations."""
    dim = 4
    mean = torch.randn(dim)
    factor = torch.randn(dim, dim)
    covariance = factor @ factor.mT + torch.eye(dim)

    with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
        log_prob(mean, (mean, covariance), parametrization="unknown")


class TestFisher:
    r"""Tests for the Fisher operator."""

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_matches_kl_curvature(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test the Fisher metric against the local KL curvature."""
        torch.manual_seed(seed)
        dim = 4

        mean = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)
        delta_mean = torch.randn(*batch_shape, dim)
        chol = torch.linalg.cholesky(covariance)
        match parametrization:
            case CovarianceType.COVARIANCE:
                theta = (mean, covariance)
                tangent = (
                    delta_mean,
                    _symmetric(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.PRECISION:
                theta = (mean, torch.linalg.inv(covariance))
                tangent = (
                    delta_mean,
                    _symmetric(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.CHOLESKY:
                theta = (mean, chol)
                tangent = (
                    delta_mean,
                    torch.tril(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.LOG_CHOLESKY:
                theta = (
                    mean,
                    chol.tril(diagonal=-1)
                    + torch.diag_embed(chol.diagonal(dim1=-2, dim2=-1).log()),
                )
                tangent = (
                    delta_mean,
                    torch.tril(torch.randn(*batch_shape, dim, dim)),
                )

        expected = _parameter_inner_product(
            tangent,
            fisher(theta, tangent, parametrization=parametrization),
        ).sum()
        actual_tangent = fisher(theta, tangent, parametrization=parametrization)
        actual = _directional_second_derivative(
            lambda t: kl(
                (mean + t * tangent[0], theta[1] + t * tangent[1]),
                theta,
                parametrization=parametrization,
            ).sum()
        )

        assert actual_tangent[0].shape == (*batch_shape, dim)
        assert actual_tangent[1].shape == (*batch_shape, dim, dim)
        assert torch.allclose(actual, expected)

    def test_rejects_unknown_parametrization(self) -> None:
        r"""Test that the public Fisher dispatch rejects unknown parametrizations."""
        dim = 4
        mean = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)

        with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
            fisher((mean, covariance), (mean, covariance), parametrization="unknown")

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_inverse_fisher(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test that the inverse Fisher operator inverts the Fisher operator."""
        torch.manual_seed(seed)
        dim = 4

        mean = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)

        match parametrization:
            case CovarianceType.COVARIANCE:
                theta = (mean, covariance)
                tangent = (
                    torch.randn(*batch_shape, dim),
                    _symmetric(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.PRECISION:
                theta = (mean, torch.linalg.inv(covariance))
                tangent = (
                    torch.randn(*batch_shape, dim),
                    _symmetric(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.CHOLESKY:
                theta = (mean, torch.linalg.cholesky(covariance))
                tangent = (
                    torch.randn(*batch_shape, dim),
                    torch.tril(torch.randn(*batch_shape, dim, dim)),
                )
            case CovarianceType.LOG_CHOLESKY:
                chol = torch.linalg.cholesky(covariance)
                theta = (
                    mean,
                    chol.tril(diagonal=-1)
                    + torch.diag_embed(chol.diagonal(dim1=-2, dim2=-1).log()),
                )
                tangent = (
                    torch.randn(*batch_shape, dim),
                    torch.tril(torch.randn(*batch_shape, dim, dim)),
                )
            case _:
                raise AssertionError(f"Unexpected parametrization {parametrization!r}.")

        transported = fisher(theta, tangent, parametrization=parametrization)
        recovered = inverse_fisher(theta, transported, parametrization=parametrization)

        assert transported[0].shape == (*batch_shape, dim)
        assert transported[1].shape == (*batch_shape, dim, dim)
        assert recovered[0].shape == (*batch_shape, dim)
        assert recovered[1].shape == (*batch_shape, dim, dim)
        assert (recovered[0] - tangent[0]).abs().amax() < 1e-5
        assert (recovered[1] - tangent[1]).abs().amax() < 1e-5

    def test_inverse_fisher_rejects_unknown_parametrization(self) -> None:
        r"""Test that the public inverse Fisher dispatch rejects unknown parametrizations."""
        dim = 4
        mean = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)

        with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
            inverse_fisher(
                (mean, covariance),
                (mean, covariance),
                parametrization="unknown",
            )


class TestArgminProximalKL:
    r"""Tests for the KL-proximal Gaussian update."""

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_solve_proximal_kl_supports_batched_theta(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test the direct KL-proximal solver on vectorized linear gradients."""
        torch.manual_seed(seed)
        dim = 4
        gamma = torch.tensor(1.7)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        cov_prior = factor @ factor.mT + torch.eye(dim)
        prec_prior = torch.cholesky_inverse(torch.linalg.cholesky(cov_prior))
        g = torch.randn(*batch_shape, dim)
        expected_mean = mean_prior - torch.einsum(
            "...ij,...j->...i", cov_prior, g / gamma
        )
        prec_chol_prior = torch.linalg.cholesky(prec_prior)
        chol_prior = torch.linalg.cholesky(cov_prior)
        log_chol_prior = chol_prior.tril(diagonal=-1) + torch.diag_embed(
            chol_prior.diagonal(dim1=-2, dim2=-1).log()
        )

        match parametrization:
            case CovarianceType.COVARIANCE:
                prec_shift_factor = torch.randn(*batch_shape, dim, dim)
                prec_shift = prec_shift_factor @ prec_shift_factor.mT
                grad_mat = 0.5 * gamma * prec_shift
                theta_prior = (mean_prior, cov_prior)
                expected_prec = 0.5 * (
                    prec_prior + prec_shift + (prec_prior + prec_shift).mT
                )
                expected_matrix = torch.cholesky_inverse(
                    torch.linalg.cholesky(expected_prec)
                )
                matrix_tol = 1e-4
                mean_tol = 1e-6
                projected_grad = _symmetric

            case CovarianceType.PRECISION:
                grad_prec_factor = torch.randn(*batch_shape, dim, dim)
                grad_mat = grad_prec_factor @ grad_prec_factor.mT
                theta_prior = (mean_prior, prec_prior)
                white_grad = prec_chol_prior.mT @ grad_mat @ prec_chol_prior
                white_grad = _symmetric(white_grad)
                eigvals, eigvecs = torch.linalg.eigh(white_grad)
                spectral_scale = 2 / (1 + torch.sqrt(1 + 8 * eigvals / gamma))
                white_prec = eigvecs @ torch.diag_embed(spectral_scale) @ eigvecs.mT
                expected_matrix = prec_chol_prior @ white_prec @ prec_chol_prior.mT
                matrix_tol = 1e-4
                mean_tol = 1e-5
                projected_grad = _symmetric

            case CovarianceType.CHOLESKY:
                grad_mat = torch.tril(torch.randn(*batch_shape, dim, dim))
                theta_prior = (mean_prior, chol_prior)
                white_grad = chol_prior.mT @ grad_mat
                diag_grad = white_grad.diagonal(dim1=-2, dim2=-1)
                diag_update = (
                    0.5
                    * (-diag_grad + torch.sqrt(diag_grad.square() + 4 * gamma.square()))
                    / gamma
                )
                white_chol = torch.tril(
                    -white_grad / gamma, diagonal=-1
                ) + torch.diag_embed(diag_update)
                expected_matrix = torch.tril(chol_prior @ white_chol)
                matrix_tol = 1e-5
                mean_tol = 1e-5
                projected_grad = torch.tril

            case CovarianceType.LOG_CHOLESKY:
                grad_mat = torch.tril(torch.randn(*batch_shape, dim, dim))
                diag_grad = grad_mat.diagonal(dim1=-2, dim2=-1)
                diag_grad = 0.5 * gamma * torch.tanh(diag_grad)
                grad_mat = grad_mat.tril(diagonal=-1) + torch.diag_embed(diag_grad)
                theta_prior = (mean_prior, log_chol_prior)
                grad_off = torch.tril(grad_mat, diagonal=-1)
                lin = chol_prior.mT @ grad_off
                diag_lin = lin.diagonal(dim1=-2, dim2=-1)
                diag_update = (
                    0.5
                    * (
                        -diag_lin
                        + torch.sqrt(
                            diag_lin.square() + 4 * gamma * (gamma - diag_grad)
                        )
                    )
                    / gamma
                )
                white_chol = torch.tril(-lin / gamma, diagonal=-1) + torch.diag_embed(
                    diag_update
                )
                expected_chol = torch.tril(chol_prior @ white_chol)
                expected_matrix = expected_chol.tril(diagonal=-1) + torch.diag_embed(
                    expected_chol.diagonal(dim1=-2, dim2=-1).log()
                )
                matrix_tol = 1e-5
                mean_tol = 1e-5
                projected_grad = torch.tril

        def grad_fn(theta: tuple[Tensor, Tensor], /) -> tuple[Tensor, Tensor]:
            mean, matrix = theta
            assert mean.shape == (*batch_shape, dim)
            assert matrix.shape == (*batch_shape, dim, dim)
            return g, grad_mat

        mean_post, matrix_post = solve_proximal_reverse_kl(
            grad_fn,
            theta_prior,
            gamma=gamma,
            parametrization=parametrization,
        )

        assert mean_post.shape == (*batch_shape, dim)
        assert matrix_post.shape == (*batch_shape, dim, dim)
        assert (mean_post - expected_mean).abs().amax() < 1e-5
        assert (matrix_post - expected_matrix).abs().amax() < 1e-5

        mean_var = mean_post.detach().clone().requires_grad_(True)
        matrix_var = matrix_post.detach().clone().requires_grad_(True)
        objective = (
            (g * (mean_var - mean_prior)).sum()
            + (grad_mat * (matrix_var - theta_prior[1])).sum()
            + gamma
            * kl(
                (mean_var, matrix_var),
                theta_prior,
                parametrization=parametrization,
            ).sum()
        )
        mean_grad, matrix_grad = torch.autograd.grad(
            objective,
            (mean_var, matrix_var),
        )

        assert mean_grad.shape == (*batch_shape, dim)
        assert matrix_grad.shape == (*batch_shape, dim, dim)
        assert mean_grad.abs().amax() < mean_tol
        assert projected_grad(matrix_grad).abs().amax() < matrix_tol

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_solves_closed_form_problem(
        self, seed: int, parametrization: CovarianceType
    ) -> None:
        r"""Test the KL-proximal Gaussian update."""
        torch.manual_seed(seed)
        batch_shape = (2, 3)
        dim = 4
        gamma = torch.tensor(1.7)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        precision_prior = torch.cholesky_inverse(
            torch.linalg.cholesky(covariance_prior)
        )
        g = torch.randn(*batch_shape, dim)
        expected_mean = mean_prior - torch.einsum(
            "...ij,...j->...i", covariance_prior, g / gamma
        )
        precision_chol_prior = torch.linalg.cholesky(precision_prior)
        covariance_chol_prior = torch.linalg.cholesky(covariance_prior)
        log_chol_prior = covariance_chol_prior.tril(diagonal=-1) + torch.diag_embed(
            covariance_chol_prior.diagonal(dim1=-2, dim2=-1).log()
        )
        match parametrization:
            case CovarianceType.COVARIANCE:
                precision_shift_factor = torch.randn(*batch_shape, dim, dim)
                precision_shift = precision_shift_factor @ precision_shift_factor.mT
                gradient_matrix = 0.5 * gamma * precision_shift
                theta_prior = (mean_prior, covariance_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    mean, covariance = theta
                    return (g * mean).sum() + (gradient_matrix * covariance).sum()

                expected_precision = 0.5 * (
                    precision_prior
                    + precision_shift
                    + (precision_prior + precision_shift).mT
                )
                expected_matrix = torch.cholesky_inverse(
                    torch.linalg.cholesky(expected_precision)
                )
                matrix_tol = 1e-4
                mean_tol = 1e-6
                projected_grad = _symmetric

            case CovarianceType.PRECISION:
                gradient_precision_factor = torch.randn(*batch_shape, dim, dim)
                gradient_matrix = (
                    gradient_precision_factor @ gradient_precision_factor.mT
                )
                theta_prior = (mean_prior, precision_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    mean, precision = theta
                    return (g * mean).sum() + (gradient_matrix * precision).sum()

                whitened_gradient = (
                    precision_chol_prior.mT @ gradient_matrix @ precision_chol_prior
                )
                whitened_gradient = _symmetric(whitened_gradient)
                eigenvalues, eigenvectors = torch.linalg.eigh(whitened_gradient)
                spectral_scale = 2 / (1 + torch.sqrt(1 + 8 * eigenvalues / gamma))
                whitened_precision = (
                    eigenvectors @ torch.diag_embed(spectral_scale) @ eigenvectors.mT
                )
                expected_matrix = (
                    precision_chol_prior @ whitened_precision @ precision_chol_prior.mT
                )
                matrix_tol = 1e-4
                mean_tol = 1e-5
                projected_grad = _symmetric

            case CovarianceType.CHOLESKY:
                gradient_matrix = torch.tril(torch.randn(*batch_shape, dim, dim))
                theta_prior = (mean_prior, covariance_chol_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    mean, chol = theta
                    return (g * mean).sum() + (gradient_matrix * chol).sum()

                whitened_gradient = covariance_chol_prior.mT @ gradient_matrix
                diagonal_gradient = whitened_gradient.diagonal(dim1=-2, dim2=-1)
                diagonal_update = (
                    0.5
                    * (
                        -diagonal_gradient
                        + torch.sqrt(diagonal_gradient.square() + 4 * gamma.square())
                    )
                    / gamma
                )
                whitened_cholesky = torch.tril(
                    -whitened_gradient / gamma, diagonal=-1
                ) + torch.diag_embed(diagonal_update)
                expected_matrix = torch.tril(covariance_chol_prior @ whitened_cholesky)
                matrix_tol = 1e-5
                mean_tol = 1e-5
                projected_grad = torch.tril

            case CovarianceType.LOG_CHOLESKY:
                gradient_matrix = torch.tril(torch.randn(*batch_shape, dim, dim))
                diagonal_gradient = gradient_matrix.diagonal(dim1=-2, dim2=-1)
                diagonal_gradient = 0.5 * gamma * torch.tanh(diagonal_gradient)
                gradient_matrix = gradient_matrix.tril(diagonal=-1) + torch.diag_embed(
                    diagonal_gradient
                )
                theta_prior = (mean_prior, log_chol_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    mean, log_chol = theta
                    return (g * mean).sum() + (gradient_matrix * log_chol).sum()

                gradient_off = torch.tril(gradient_matrix, diagonal=-1)
                linear_term = covariance_chol_prior.mT @ gradient_off
                diagonal_linear = linear_term.diagonal(dim1=-2, dim2=-1)
                diagonal_update = (
                    0.5
                    * (
                        -diagonal_linear
                        + torch.sqrt(
                            diagonal_linear.square()
                            + 4 * gamma * (gamma - diagonal_gradient)
                        )
                    )
                    / gamma
                )
                whitened_cholesky = torch.tril(
                    -linear_term / gamma, diagonal=-1
                ) + torch.diag_embed(diagonal_update)
                expected_cholesky = torch.tril(
                    covariance_chol_prior @ whitened_cholesky
                )
                expected_matrix = expected_cholesky.tril(
                    diagonal=-1
                ) + torch.diag_embed(expected_cholesky.diagonal(dim1=-2, dim2=-1).log())
                matrix_tol = 1e-5
                mean_tol = 1e-5
                projected_grad = torch.tril

        mean_post, matrix_post = argmin_proximal_reverse_kl(
            objective_fn,
            theta_prior,
            gamma=gamma,
            parametrization=parametrization,
        )

        assert (mean_post - expected_mean).abs().amax() < 1e-5
        assert (matrix_post - expected_matrix).abs().amax() < 1e-5

        mean_var = mean_post.detach().clone().requires_grad_(True)
        matrix_var = matrix_post.detach().clone().requires_grad_(True)
        objective = (
            (g * (mean_var - mean_prior)).sum()
            + (gradient_matrix * (matrix_var - theta_prior[1])).sum()
            + gamma
            * kl(
                (mean_var, matrix_var),
                theta_prior,
                parametrization=parametrization,
            ).sum()
        )
        mean_grad, matrix_grad = torch.autograd.grad(
            objective,
            (mean_var, matrix_var),
        )

        assert mean_grad.abs().amax() < mean_tol
        assert projected_grad(matrix_grad).abs().amax() < matrix_tol

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_raises_when_objective_has_no_finite_minimizer(
        self, seed: int, parametrization: CovarianceType
    ) -> None:
        r"""Test that ill-posed linearized objectives are rejected."""
        torch.manual_seed(seed)
        dim = 4
        mean_prior = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        precision_prior = torch.linalg.inv(covariance_prior)
        chol_prior = torch.linalg.cholesky(covariance_prior)
        log_chol_prior = chol_prior.tril(diagonal=-1) + torch.diag_embed(
            chol_prior.diagonal(dim1=-2, dim2=-1).log()
        )
        mean_gradient = torch.randn(dim)
        gamma = torch.tensor(1.0)

        match parametrization:
            case CovarianceType.COVARIANCE:
                theta_prior = (mean_prior, covariance_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    return (mean_gradient * theta[0]).sum() + (
                        -precision_prior * theta[1]
                    ).sum()

            case CovarianceType.PRECISION:
                theta_prior = (mean_prior, precision_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    return (mean_gradient * theta[0]).sum() + (
                        -torch.eye(dim) * theta[1]
                    ).sum()

            case CovarianceType.CHOLESKY:
                pytest.skip("Not implemented")

            case CovarianceType.LOG_CHOLESKY:
                diagonal_gradient = gamma + 0.1 + torch.rand(dim)
                gradient_log_cholesky = torch.diag_embed(diagonal_gradient)
                theta_prior = (mean_prior, log_chol_prior)

                def objective_fn(theta: tuple[Tensor, Tensor], /) -> Tensor:
                    return (mean_gradient * theta[0]).sum() + (
                        gradient_log_cholesky * theta[1]
                    ).sum()

        with pytest.raises(ValueError, match="finite minimizer"):
            argmin_proximal_reverse_kl(
                objective_fn,
                theta_prior,
                gamma=gamma,
                parametrization=parametrization,
            )

    def test_rejects_unknown_parametrization(self) -> None:
        r"""Test that the proximal update rejects unknown parametrizations."""
        dim = 4
        mean = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)

        with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
            argmin_proximal_reverse_kl(
                lambda theta: theta[0].sum() + theta[1].sum(),
                (mean, covariance),
                gamma=2.0,
                parametrization="unknown",
            )


class TestArgminForwardKL:
    r"""Tests for the exact forward-KL Gaussian update."""

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("rho_mode", RHO_MODES, ids="rho={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_matches_covariance_branch(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
        rho_mode: str,
    ) -> None:
        r"""Test that all parametrizations agree with the covariance update."""
        torch.manual_seed(seed)
        dim = 4
        rho = _make_rho(batch_shape, rho_mode)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)

        expected = argmin_forward_kl(
            z_obs,
            (mean_prior, covariance_prior),
            retention=rho,
            parametrization="covariance",
        )
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        actual = argmin_forward_kl(
            z_obs,
            theta_prior,
            retention=rho,
            parametrization=parametrization,
        )
        actual_covariance = parametrization.to_covariance(actual)

        assert (actual_covariance[0] - expected[0]).abs().amax() < 1e-5
        assert (actual_covariance[1] - expected[1]).abs().amax() < 1e-5

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_matches_closed_form(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test the exact forward-KL update across batch shapes."""
        torch.manual_seed(seed)
        dim = 4
        rho = torch.tensor(1.7 / 2.7)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)
        delta = z_obs - mean_prior
        outer = torch.einsum("...i, ...j -> ...ij", delta, delta)
        expected_mean = mean_prior + (1 - rho) * delta
        expected_covariance = rho * covariance_prior + (1 - rho) * rho * outer

        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        actual = argmin_forward_kl(
            z_obs,
            theta_prior,
            retention=rho,
            parametrization=parametrization,
        )
        actual_mean, actual_covariance = parametrization.to_covariance(actual)

        assert actual[0].shape == (*batch_shape, dim)
        assert actual[1].shape == (*batch_shape, dim, dim)
        assert (actual_mean - expected_mean).abs().amax() < 1e-5
        assert (actual_covariance - expected_covariance).abs().amax() < 1e-5

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_stationary_for_exact_objective(
        self, seed: int, parametrization: CovarianceType
    ) -> None:
        r"""Test that the returned point is stationary for the exact objective."""
        torch.manual_seed(seed)
        batch_shape = (2, 3)
        dim = 4
        rho = torch.tensor(1.7 / 2.7)
        gamma = rho / (1 - rho)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        theta_post = argmin_forward_kl(
            z_obs,
            theta_prior,
            retention=rho,
            parametrization=parametrization,
        )
        mean_var = theta_post[0].detach().clone().requires_grad_(True)
        matrix_var = theta_post[1].detach().clone().requires_grad_(True)
        objective = (
            -log_prob(z_obs, (mean_var, matrix_var), parametrization=parametrization)
            + gamma
            * kl(theta_prior, (mean_var, matrix_var), parametrization=parametrization)
        ).sum()
        mean_grad, matrix_grad = torch.autograd.grad(objective, (mean_var, matrix_var))

        match parametrization:
            case CovarianceType.COVARIANCE | CovarianceType.PRECISION:
                projected_grad = _symmetric(matrix_grad)
            case CovarianceType.CHOLESKY | CovarianceType.LOG_CHOLESKY:
                projected_grad = torch.tril(matrix_grad)

        assert mean_grad.abs().amax() < 1e-5
        assert projected_grad.abs().amax() < 1e-5

    @pytest.mark.parametrize("rho_mu", [0.0, 1e-8, 1e-3, 0.5, 1.0])
    @pytest.mark.parametrize("rho_sigma", [1e-8, 1e-3, 0.5, 1.0 - 1e-8, 1.0])
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_grad_finite_at_zero_innovation(
        self,
        parametrization: CovarianceType,
        rho_mu: float,
        rho_sigma: float,
    ) -> None:
        r"""Test that zero-innovation gradients stay finite down to $ρ_Σ \to 0⁺$."""
        dim = 3
        z = torch.zeros(1, dim, dtype=torch.float64, requires_grad=True)
        mean_prior = torch.zeros(1, dim, dtype=torch.float64, requires_grad=True)
        covariance_prior = (
            torch.eye(dim, dtype=torch.float64).expand(1, dim, dim).clone()
        )
        theta_prior = parametrization.from_covariance(
            (mean_prior.detach(), covariance_prior)
        )
        matrix_prior = theta_prior[1].detach().clone().requires_grad_(True)
        rho_mu_tensor = torch.tensor(rho_mu, dtype=torch.float64, requires_grad=True)
        rho_sigma_tensor = torch.tensor(
            rho_sigma, dtype=torch.float64, requires_grad=True
        )

        mean_post, matrix_post = argmin_forward_kl(
            z,
            (mean_prior, matrix_prior),
            retention=(rho_mu_tensor, rho_sigma_tensor),
            parametrization=parametrization,
        )
        (mean_post.sum() + matrix_post.sum()).backward()

        for grad in (
            z.grad,
            mean_prior.grad,
            matrix_prior.grad,
            rho_mu_tensor.grad,
            rho_sigma_tensor.grad,
        ):
            assert grad is not None
            assert torch.isfinite(grad).all()

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_rho_one_converges_to_identity(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test that $ρ = 1$ gives the identity update."""
        torch.manual_seed(seed)
        dim = 4
        rho = torch.tensor(1.0)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        mean_post, covariance_post = parametrization.to_covariance(
            argmin_forward_kl(
                z_obs,
                theta_prior,
                retention=rho,
                parametrization=parametrization,
            ),
        )

        assert (mean_post - mean_prior).abs().amax() < 1e-5
        assert (covariance_post - covariance_prior).abs().amax() < 1e-4

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_rho_near_zero_converges_to_observation(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test that $ρ \to 0$ nearly collapses onto the observation."""
        torch.manual_seed(seed)
        dim = 4
        rho = torch.tensor(1e-6)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        mean_post, covariance_post = parametrization.to_covariance(
            argmin_forward_kl(
                z_obs,
                theta_prior,
                retention=rho,
                parametrization=parametrization,
            ),
        )

        assert (mean_post - z_obs).abs().amax() < 1e-5
        assert covariance_post.abs().amax() < 5e-5

    def test_rejects_unknown_parametrization(self) -> None:
        r"""Test that the forward-KL update rejects unknown parametrizations."""
        dim = 4
        mean = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(dim)

        with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
            argmin_forward_kl(
                z_obs,
                (mean, covariance),
                retention=0.5,
                parametrization="unknown",
            )

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_compile_fullgraph(self, parametrization: CovarianceType) -> None:
        r"""Test that the exact forward-KL update compiles with `fullgraph=True`."""
        dim = 4
        rho = torch.tensor(1.7 / 2.7)
        mean_prior = torch.randn(2, dim)
        factor = torch.randn(2, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        z_obs = torch.randn(2, dim)

        def update(z: Tensor, theta: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            return argmin_forward_kl(
                z,
                theta,
                retention=rho,
                parametrization=parametrization,
            )

        compiled = torch.compile(update, fullgraph=True)

        expected = update(z_obs, theta_prior)
        actual = compiled(z_obs, theta_prior)

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


class TestArgminReverseKL:
    r"""Tests for the exact reverse-KL Gaussian update."""

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("rho_mode", RHO_MODES, ids="rho={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_matches_covariance_branch(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
        rho_mode: str,
    ) -> None:
        r"""Test that all parametrizations agree with the covariance update."""
        torch.manual_seed(seed)
        dim = 4
        rho = _make_reverse_rho(batch_shape, rho_mode)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)

        expected = argmin_reverse_kl(
            z_obs,
            (mean_prior, covariance_prior),
            retention=rho,
            parametrization="covariance",
        )
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        actual = argmin_reverse_kl(
            z_obs,
            theta_prior,
            retention=rho,
            parametrization=parametrization,
        )
        actual_covariance = parametrization.to_covariance(actual)

        assert (actual_covariance[0] - expected[0]).abs().amax() < 1e-5
        assert (actual_covariance[1] - expected[1]).abs().amax() < 1e-5

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_matches_closed_form(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test the exact reverse-KL update against the whitened closed form."""
        torch.manual_seed(seed)
        dim = 4
        gamma = torch.tensor(1.7)
        retention = (gamma / (1.0 + gamma), (gamma - 1.0) / gamma)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)
        delta = z_obs - mean_prior
        prior_chol = torch.linalg.cholesky(covariance_prior)
        whitened_delta = torch.linalg.solve_triangular(
            prior_chol,
            delta.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        q = (whitened_delta * whitened_delta).sum(dim=-1)
        beta = (gamma - 1) / gamma
        s_parallel = _solve_reverse_kl_bisection(q, gamma)
        mean_scale = (1 + gamma * s_parallel).reciprocal()
        outer = torch.einsum("...i, ...j -> ...ij", delta, delta)
        coefficient = torch.where(q > 0, (s_parallel - beta) / q, torch.zeros_like(q))
        expected_mean = mean_prior + mean_scale[..., None] * delta
        expected_covariance = (
            beta[..., None, None] * covariance_prior
            + coefficient[..., None, None] * outer
        )

        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        actual = argmin_reverse_kl(
            z_obs,
            theta_prior,
            retention=retention,
            parametrization=parametrization,
        )
        actual_mean, actual_covariance = parametrization.to_covariance(actual)

        assert actual[0].shape == (*batch_shape, dim)
        assert actual[1].shape == (*batch_shape, dim, dim)
        assert (actual_mean - expected_mean).abs().amax() < 1e-5
        assert (actual_covariance - expected_covariance).abs().amax() < 1e-5

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_stationary_for_exact_objective(
        self, seed: int, parametrization: CovarianceType
    ) -> None:
        r"""Test that the returned point is stationary for the exact objective."""
        torch.manual_seed(seed)
        batch_shape = (2, 3)
        dim = 4
        gamma = torch.tensor(1.7)
        retention = (gamma / (1.0 + gamma), (gamma - 1.0) / gamma)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        theta_post = argmin_reverse_kl(
            z_obs,
            theta_prior,
            retention=retention,
            parametrization=parametrization,
        )
        mean_var = theta_post[0].detach().clone().requires_grad_(True)
        matrix_var = theta_post[1].detach().clone().requires_grad_(True)
        objective = (
            -log_prob(z_obs, (mean_var, matrix_var), parametrization=parametrization)
            + gamma
            * kl((mean_var, matrix_var), theta_prior, parametrization=parametrization)
        ).sum()
        mean_grad, matrix_grad = torch.autograd.grad(objective, (mean_var, matrix_var))

        match parametrization:
            case CovarianceType.COVARIANCE | CovarianceType.PRECISION:
                projected_grad = _symmetric(matrix_grad)
            case CovarianceType.CHOLESKY | CovarianceType.LOG_CHOLESKY:
                projected_grad = torch.tril(matrix_grad)

        assert mean_grad.abs().amax() < 1e-5
        assert projected_grad.abs().amax() < 1e-5

    @pytest.mark.parametrize(
        "rho_mu",
        [0.0, 1e-8, 1e-5, 1e-3, 0.5, 10.0 / 11.0, 1000.0 / 1001.0, 1.0],
    )
    @pytest.mark.parametrize("rho_sigma", [1e-8, 1e-6, 1.0 / 3.0, 0.9, 1.0])
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_grad_finite_at_zero_innovation(
        self,
        parametrization: CovarianceType,
        rho_mu: float,
        rho_sigma: float,
    ) -> None:
        r"""Test that zero-innovation gradients stay finite across retention endpoints."""
        dim = 3
        z = torch.zeros(1, dim, dtype=torch.float64, requires_grad=True)
        mean_prior = torch.zeros(1, dim, dtype=torch.float64, requires_grad=True)
        covariance_prior = (
            torch.eye(dim, dtype=torch.float64).expand(1, dim, dim).clone()
        )
        theta_prior = parametrization.from_covariance(
            (mean_prior.detach(), covariance_prior)
        )
        matrix_prior = theta_prior[1].detach().clone().requires_grad_(True)
        rho_mu_tensor = torch.tensor(rho_mu, dtype=torch.float64, requires_grad=True)
        rho_sigma_tensor = torch.tensor(
            rho_sigma, dtype=torch.float64, requires_grad=True
        )

        mean_post, matrix_post = argmin_reverse_kl(
            z,
            (mean_prior, matrix_prior),
            retention=(rho_mu_tensor, rho_sigma_tensor),
            parametrization=parametrization,
        )
        (mean_post.sum() + matrix_post.sum()).backward()

        for grad in (
            z.grad,
            mean_prior.grad,
            matrix_prior.grad,
            rho_mu_tensor.grad,
            rho_sigma_tensor.grad,
        ):
            assert grad is not None
            assert torch.isfinite(grad).all()

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_retention_one_converges_to_identity(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test that retention $(1, 1)$ gives the identity update."""
        torch.manual_seed(seed)
        dim = 4
        retention = (torch.tensor(1.0), torch.tensor(1.0))

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        mean_post, covariance_post = parametrization.to_covariance(
            argmin_reverse_kl(
                z_obs,
                theta_prior,
                retention=retention,
                parametrization=parametrization,
            ),
        )

        assert (mean_post - mean_prior).abs().amax() < 1e-5
        assert (covariance_post - covariance_prior).abs().amax() < 2e-5

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids="batch_shape={}".format)
    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_rho_sigma_at_most_zero_rejects(
        self,
        seed: int,
        parametrization: CovarianceType,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Test that the exact reverse-KL update rejects non-admissible retention."""
        torch.manual_seed(seed)
        dim = 4
        retention = (0.5, 0.0)

        mean_prior = torch.randn(*batch_shape, dim)
        factor = torch.randn(*batch_shape, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(*batch_shape, dim)
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))

        with pytest.raises(AssertionError, match="requires rho_sigma in \\(0, 1\\]"):
            argmin_reverse_kl(
                z_obs,
                theta_prior,
                retention=retention,
                parametrization=parametrization,
            )

    def test_rejects_unknown_parametrization(self) -> None:
        r"""Test that the reverse-KL update rejects unknown parametrizations."""
        dim = 4
        mean = torch.randn(dim)
        factor = torch.randn(dim, dim)
        covariance = factor @ factor.mT + torch.eye(dim)
        z_obs = torch.randn(dim)

        with pytest.raises(ValueError, match="'unknown' is not a valid CovarianceType"):
            argmin_reverse_kl(
                z_obs,
                (mean, covariance),
                retention=(0.5, 0.5),
                parametrization="unknown",
            )

    @pytest.mark.parametrize("parametrization", CovarianceType)
    def test_compile_fullgraph(self, parametrization: CovarianceType) -> None:
        r"""Test that the exact reverse-KL update compiles with `fullgraph=True`."""
        dim = 4
        gamma = torch.tensor(1.7)
        retention = (gamma / (1.0 + gamma), (gamma - 1.0) / gamma)
        mean_prior = torch.randn(2, dim)
        factor = torch.randn(2, dim, dim)
        covariance_prior = factor @ factor.mT + torch.eye(dim)
        theta_prior = parametrization.from_covariance((mean_prior, covariance_prior))
        z_obs = torch.randn(2, dim)

        def update(z: Tensor, theta: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            return argmin_reverse_kl(
                z,
                theta,
                retention=retention,
                parametrization=parametrization,
            )

        compiled = torch.compile(update, fullgraph=True)

        expected = update(z_obs, theta_prior)
        actual = compiled(z_obs, theta_prior)

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
