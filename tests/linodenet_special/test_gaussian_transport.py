r"""Tests for transport maps."""

import math

import pytest
import torch
from torch import Tensor
from torch.autograd import gradcheck

from linodenet_special import hard_bend
from linodenet_special.compiled import (
    bimodal_to_gaussian as bimodal_to_gaussian_cpp,
    gaussian_to_bimodal as gaussian_to_bimodal_cpp,
    gaussian_to_mixture as gaussian_to_mixture_cpp,
    mixture_to_gaussian as mixture_to_gaussian_cpp,
)
from linodenet_special.fallbacks import (
    bimodal_to_gaussian as bimodal_to_gaussian_py,
    bimodal_to_gaussian_value_and_grad as bimodal_to_gaussian_value_and_jac_py,
    gaussian_to_bimodal as gaussian_to_bimodal_py,
    gaussian_to_bimodal_value_and_grad as gaussian_to_bimodal_value_and_jac_py,
    gaussian_to_mixture as gaussian_to_mixture_py,
    gaussian_to_mixture_value_and_grad as gaussian_to_mixture_value_and_jac_py,
    mixture_to_gaussian as mixture_to_gaussian_py,
    mixture_to_gaussian_value_and_grad as mixture_to_gaussian_value_and_jac_py,
)
from linodenet_special.interfaces import (
    BimodalToGaussian,
    GaussianToBimodal,
    GaussianToMixture,
    MixtureToGaussian,
)
from tests.testing import DEVICES, DTYPES, TestCase

BIMODAL_TO_GAUSSIAN: dict[str, BimodalToGaussian] = {
    "cpp": bimodal_to_gaussian_cpp,
    "py": bimodal_to_gaussian_py,
}
GAUSSIAN_TO_BIMODAL: dict[str, GaussianToBimodal] = {
    "cpp": gaussian_to_bimodal_cpp,
    "py": gaussian_to_bimodal_py,
}
MIXTURE_TO_GAUSSIAN: dict[str, MixtureToGaussian] = {
    "cpp": mixture_to_gaussian_cpp,
    "py": mixture_to_gaussian_py,
}
GAUSSIAN_TO_MIXTURE: dict[str, GaussianToMixture] = {
    "cpp": gaussian_to_mixture_cpp,
    "py": gaussian_to_mixture_py,
}
BIMODAL_TO_GAUSSIAN_VALUE_AND_JAC = {
    "py": bimodal_to_gaussian_value_and_jac_py,
}
GAUSSIAN_TO_BIMODAL_VALUE_AND_JAC = {
    "py": gaussian_to_bimodal_value_and_jac_py,
}
MIXTURE_TO_GAUSSIAN_VALUE_AND_JAC = {
    "py": mixture_to_gaussian_value_and_jac_py,
}
GAUSSIAN_TO_MIXTURE_VALUE_AND_JAC = {
    "py": gaussian_to_mixture_value_and_jac_py,
}


class BimodalTest(TestCase):
    SEED = 0
    N = 256
    X_MIN = -20
    X_MAX = 20

    STDVS = [0.25, 0.5, 1, 2, 10]
    MEANS = [0.1, 0.5, 1, 2, 4]
    TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-7, 1e-7),
    }
    FORWARD_GRADCHECK_TOL = {
        torch.float32: (1e-3, 1e-3, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }
    INVERSE_GRADCHECK_TOL = {
        torch.float32: (1e-2, 1e-2, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    SAFE_SIGMA_THRESHOLD = {
        torch.float32: 3.0,
        torch.float64: 5.0,
    }

    @staticmethod
    def get_x_star(mean: Tensor, stdv: Tensor) -> Tensor:
        r"""$x⁎ = Ψ⁻¹'(0) = σ⋅exp(½μ²/σ²)$."""
        return stdv * math.exp(0.5 * (mean / stdv) ** 2)

    @staticmethod
    def get_y_star(mean: Tensor, stdv: Tensor) -> Tensor:
        r"""$y⁎ = Ψ'(0) = σ⁻¹exp(-½μ²/σ²)$."""
        return math.exp(-0.5 * (mean / stdv) ** 2) / stdv

    @classmethod
    def get_x_safe(cls, mean: float, stdv: float, *, dtype: torch.dtype) -> float:
        r"""Return the inner cutoff for numerically stable bimodal tests.

        The slope at the origin is $Ψ'(0)=σ⁻¹ℯ^{-½(μ/σ)²}$. For a given floating
        point dtype we treat the central region as numerically flat once the
        local slope drops below $√ρ/σ$, where $ρ$ is the decimal resolution.
        Solving the corresponding Gaussian tail model yields the inner cutoff

        .. math:: x_\text{safe} = \max(0, μ - σ\sqrt{-\log ρ}).

        If $μ/σ$ is smaller than the dtype-dependent threshold $√{-\log ρ}$, we
        keep the full interval $[-μ, μ]$, so $x_\text{safe}=0$.
        """
        # sqrt(-log(resolution)) is about 3.7 for float32 and 5.9 for float64.
        threshold = math.floor(math.sqrt(-math.log(torch.finfo(dtype).resolution)))
        if mean / stdv <= threshold:
            return 0.0
        return max(0.0, mean - stdv * threshold)  # ≤ μ

    @classmethod
    def get_y_safe(cls, mean: float, stdv: float, *, dtype: torch.dtype) -> float:
        r"""Return the inner cutoff for numerically stable inverse bimodal tests.

        The inverse slope at the origin is

        .. math:: (Ψ⁻¹)'(0)=σℯ^{½(μ/σ)²}.

        We treat the center as numerically stiff once this amplification exceeds
        $1/√ρ$, where $ρ$ is the decimal resolution. Solving for the matching
        forward-side cutoff gives

        .. math:: y_\text{safe} = \max(0, (μ - x_\text{safe})/σ)

        with $x_\text{safe}$ from :meth:`get_x_safe`. If $μ/σ$ is below the
        dtype-dependent threshold, then $y_\text{safe}=0$ and we keep the full
        interval around the origin.
        """
        x_safe = cls.get_x_safe(mean, stdv, dtype=dtype)
        return max(0.0, (mean - x_safe) / stdv)  # ≤ μ/σ

    @classmethod
    def make_safe_x_range(
        cls, mean: float, stdv: float, *, dtype: torch.dtype, device: str
    ) -> Tensor:
        r"""Construct a numerically useful test range inside $[-μ, μ]$."""
        x_safe = cls.get_x_safe(mean, stdv, dtype=dtype)
        if x_safe == 0.0:
            return torch.linspace(
                *(-mean - 4 * stdv, mean + 4 * stdv),
                steps=cls.N,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        x = torch.linspace(
            *(x_safe, mean + 4 * stdv),
            steps=cls.N // 2,
            dtype=dtype,
            device=device,
        )
        return torch.cat([-x.flip(0), x]).requires_grad_(True)

    @classmethod
    def make_safe_y_range(
        cls, mean: float, stdv: float, *, dtype: torch.dtype, device: str
    ) -> Tensor:
        r"""Construct a numerically useful inverse test range around the origin."""
        y_safe = cls.get_y_safe(mean, stdv, dtype=dtype)
        if y_safe == 0.0:
            return torch.linspace(
                *(-mean / stdv - 4, mean / stdv + 4),
                steps=cls.N,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        y = torch.linspace(
            *(y_safe, mean / stdv + 4),
            steps=cls.N // 2,
            dtype=dtype,
            device=device,
        )
        return torch.cat([-y.flip(0), y]).requires_grad_(True)

    @classmethod
    def make_tail_x_range(
        cls, mean: float, stdv: float, *, dtype: torch.dtype, device: str
    ) -> Tensor:
        r"""Construct a symmetric tail range outside the bimodal transition region."""
        x_tail = torch.linspace(
            *(mean + 10 * stdv, mean + 50 * stdv),
            steps=cls.N,
            dtype=dtype,
            device=device,
        )
        return torch.cat([-x_tail.flip(0), x_tail]).requires_grad_(True)

    @classmethod
    def make_tail_y_range(
        cls, mean: float, stdv: float, *, dtype: torch.dtype, device: str
    ) -> Tensor:
        r"""Construct a symmetric inverse-tail range outside the Gaussian core."""
        y_tail = torch.linspace(
            *(mean / stdv + 10, mean / stdv + 50),
            steps=cls.N,
            dtype=dtype,
            device=device,
        )
        return torch.cat([-y_tail.flip(0), y_tail]).requires_grad_(True)

    @classmethod
    def make_full_range(cls, *, dtype: torch.dtype, device: str) -> Tensor:
        r"""Construct the shared full test range on $[X_\min, X_\max]$."""
        return torch.linspace(
            *(cls.X_MIN, cls.X_MAX),
            steps=cls.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )


class TestMixture(TestCase):
    SEED = 0
    N = 256
    CASES = [
        pytest.param(
            [0.4, 0.25, 0.35], [-1.0, 0.5, 1.5], [0.8, 1.1, 0.9], id="asymmetric"
        ),
        pytest.param([0.2, 0.5, 0.3], [-1.5, -0.5, 1.0], [1.0, 0.8, 1.2], id="shifted"),
    ]
    TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-7, 1e-7),
    }
    GRADCHECK_TOL = {
        torch.float32: (1e-3, 1e-3, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    @classmethod
    def make_full_range(
        cls, means: list[float], stdvs: list[float], *, dtype: torch.dtype, device: str
    ) -> Tensor:
        mu_min = min(means)
        mu_max = max(means)
        sigma_max = max(stdvs)
        return torch.linspace(
            *(mu_min - sigma_max, mu_max + sigma_max),
            steps=cls.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    def assert_gradcheck(
        self,
        impl,
        x: Tensor,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        *,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        omegas = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
        mus = torch.tensor(means, dtype=dtype, device=device, requires_grad=True)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device, requires_grad=True)
        atol, rtol, eps = self.GRADCHECK_TOL[dtype]
        gradcheck(
            lambda z, ω, μ, σ: impl(z, ω / ω.sum(), μ, σ),
            (x, omegas, mus, sigmas),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("stdv", BimodalTest.STDVS, ids="stdv={}".format)
@pytest.mark.parametrize("mean", BimodalTest.MEANS, ids="mean={}".format)
@pytest.mark.parametrize("name", BIMODAL_TO_GAUSSIAN, ids=str)
class TestBimodalToGaussian(BimodalTest):
    def test_special_values(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = self.get_y_star(μ, σ)

        zero = torch.tensor(0, dtype=dtype, device=device, requires_grad=True)
        y_zero = impl(zero, μ, σ)
        self.assert_close(y_zero, zero)
        y_zero.backward()
        assert zero.grad is not None
        self.assert_close(zero.grad, λ, rtol=1e-7)

    def test_forward(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)

        x = self.make_full_range(dtype=dtype, device=device)
        y = impl(x, μ, σ)
        assert y.dtype == dtype
        assert y.isfinite().all()

        self.assert_close(-y, y.flip(0))

    def test_backward(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = self.get_y_star(μ, σ)

        x = self.make_full_range(dtype=dtype, device=device)
        y = impl(x, μ, σ)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()
        self.assert_upper_bounded(x.grad, 1 / σ, rtol=0.0)
        self.assert_lower_bounded(x.grad, λ, rtol=1e-7)
        self.assert_close(x.grad, x.grad.flip(0))

    def test_gradcheck(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        x = self.make_safe_x_range(mean, stdv, dtype=dtype, device=device)

        atol, rtol, eps = self.FORWARD_GRADCHECK_TOL[dtype]
        gradcheck(
            impl,
            (x, μ, σ),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )

    def test_reversible(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = BIMODAL_TO_GAUSSIAN[name]
        inverse_impl = GAUSSIAN_TO_BIMODAL[name]
        atol, rtol = self.TOL[dtype]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        x = self.make_safe_x_range(mean, stdv, dtype=dtype, device=device)
        y = forward_impl(x, μ, σ)
        x_inv = inverse_impl(y, μ, σ)
        x_inv.sum().backward()
        assert x.grad is not None
        self.assert_close(x_inv, x, atol=atol, rtol=rtol)
        self.assert_close(x.grad, 1.0, atol=atol, rtol=rtol)

    def test_piecewise_linear_approximation(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        r"""When the gaussians are well separated, we can approximate with hard_bend."""
        impl = BIMODAL_TO_GAUSSIAN[name]
        x = self.make_full_range(dtype=dtype, device=device)
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = self.get_y_star(μ, σ)

        y = impl(x, μ, σ)
        assert y.dtype == dtype
        assert y.isfinite().all(), (
            "bimodal_to_gaussian should produce finite outputs for finite inputs"
        )

        y_approx = hard_bend(x, λ, μ / σ, 1 / σ)
        assert y_approx.dtype == dtype
        assert y_approx.isfinite().all(), (
            "Hard-contract approximation should produce finite outputs"
        )
        # y = y_approx + O(\log(2)((x-μ⋅\sign(x))/σ)⁻¹)
        atol, rtol = self.TOL[dtype]
        error = (y - y_approx).abs()
        error_bound = math.log(2) * σ / (x - x.sign() * μ).abs()
        self.assert_upper_bounded(error, error_bound, atol=atol, rtol=rtol)

    def test_tail_behavior(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        x_tail = self.make_tail_x_range(mean, stdv, dtype=dtype, device=device)
        y_tail = impl(x_tail, μ, σ)
        assert y_tail.isfinite().all()
        y_tail.sum().backward()
        assert x_tail.grad is not None
        assert x_tail.grad.isfinite().all()
        self.assert_close(x_tail.grad, 1 / σ, rtol=1e-2)
        self.assert_upper_bounded(x_tail.grad, 1 / σ, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("stdv", BimodalTest.STDVS, ids="stdv={}".format)
@pytest.mark.parametrize("mean", BimodalTest.MEANS, ids="mean={}".format)
@pytest.mark.parametrize("name", BIMODAL_TO_GAUSSIAN_VALUE_AND_JAC, ids=str)
class TestBimodalToGaussianValueAndGrad(BimodalTest):
    def test_gradcheck(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = BIMODAL_TO_GAUSSIAN_VALUE_AND_JAC[name]
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        x = self.make_safe_x_range(mean, stdv, dtype=dtype, device=device)

        atol, rtol, eps = self.FORWARD_GRADCHECK_TOL[dtype]
        gradcheck(
            impl,
            (x, μ, σ),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )

    def test_reversible(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = BIMODAL_TO_GAUSSIAN_VALUE_AND_JAC[name]
        inverse_impl = gaussian_to_bimodal_value_and_jac_py
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        x = self.make_safe_x_range(mean, stdv, dtype=dtype, device=device)

        y, d_x = forward_impl(x, μ, σ)
        x_inv, d_y = inverse_impl(y, μ, σ)
        x_inv.sum().backward()
        assert x.grad is not None

        atol, rtol = self.TOL[dtype]
        self.assert_close(x_inv, x, atol=atol, rtol=rtol)
        self.assert_close(d_x * d_y, 1.0, atol=atol, rtol=rtol)
        self.assert_close(x.grad, 1.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("stdv", BimodalTest.STDVS, ids="stdv={}".format)
@pytest.mark.parametrize("mean", BimodalTest.MEANS, ids="mean={}".format)
@pytest.mark.parametrize("name", GAUSSIAN_TO_BIMODAL, ids=str)
class TestGaussianToBimodal(BimodalTest):
    def test_special_values(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)

        zero = torch.tensor(0, dtype=dtype, device=device)
        x_zero = impl(zero, μ, σ)
        self.assert_close(x_zero, zero)

    def test_forward(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)

        y = self.make_full_range(dtype=dtype, device=device)
        x = impl(y, μ, σ)
        assert x.dtype == dtype
        assert x.isfinite().all()

        x1, x2 = x.chunk(2)
        self.assert_close(-x1, x2.flip(0))

    def test_backward(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        y_safe = self.get_y_safe(mean, stdv, dtype=dtype)
        λ_log = 0.5 * (μ / σ) ** 2 + σ.log()
        g_rtol = 2**-4
        log_tol = math.log(1 + g_rtol)
        log_grad_bound = λ_log + log_tol

        y1 = torch.linspace(
            *(y_safe, self.X_MAX),
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x1 = impl(y1, μ, σ)
        x1.sum().backward()
        assert y1.grad is not None
        assert y1.grad.isfinite().all()
        assert y1.grad.min() >= min(1, σ.item())
        assert y1.grad.log().max() <= log_grad_bound

        y2 = torch.linspace(
            *(-y_safe, self.X_MIN),
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x2 = impl(y2, μ, σ)
        x2.sum().backward()
        assert y2.grad is not None
        assert y2.grad.isfinite().all()
        assert y2.grad.min() >= min(1, σ.item())
        assert y2.grad.log().max() <= log_grad_bound
        self.assert_close(y1.grad, y2.grad)

    def test_reversible(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        inverse_impl = GAUSSIAN_TO_BIMODAL[name]
        forward_impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        y = self.make_safe_y_range(mean, stdv, dtype=dtype, device=device)
        x_inv = inverse_impl(y, μ, σ)
        y_inv = forward_impl(x_inv, μ, σ)
        y_inv.sum().backward()
        assert y.grad is not None
        atol, rtol = self.TOL[dtype]
        self.assert_close(y_inv, y, atol=atol, rtol=rtol)
        self.assert_close(y.grad, 1.0, atol=atol, rtol=rtol)

    def test_gradcheck(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        y_narrow = self.make_safe_y_range(mean, stdv, dtype=dtype, device=device)

        atol, rtol, eps = self.INVERSE_GRADCHECK_TOL[dtype]
        gradcheck(
            impl,
            (y_narrow, μ, σ),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )

    def test_piecewise_linear_approximation(
        self, name: str, mean, stdv, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL[name]
        y = self.make_full_range(dtype=dtype, device=device)
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = self.get_x_star(μ, σ)

        x = impl(y, μ, σ)
        assert x.dtype == dtype
        assert x.isfinite().all(), (
            "gaussian_to_bimodal should produce finite outputs for finite inputs"
        )

        x_approx = hard_bend(y, λ, μ, σ)
        assert x_approx.dtype == dtype
        assert x_approx.isfinite().all(), (
            "Hard-expand approximation should produce finite outputs"
        )
        # x = x_approx - \log(2)σ/y + O(y⁻³)
        atol, rtol = self.TOL[dtype]
        error = (x - x_approx).abs()
        bound = (-math.log(2) * σ / y).abs()
        self.assert_upper_bounded(error, bound, atol=atol, rtol=rtol)

    def test_tail_behavior(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = self.get_y_star(μ, σ)
        y_tail = self.make_tail_y_range(mean, stdv, dtype=dtype, device=device)
        x_tail = impl(y_tail, μ, σ)
        x_tail_approx = hard_bend(y_tail, 1 / λ, μ, σ)
        self.assert_upper_bounded((x_tail - x_tail_approx).abs(), σ / y_tail.abs())

        tail = self.make_tail_y_range(mean, stdv, dtype=dtype, device=device)
        x_tail = impl(tail, μ, σ)
        assert x_tail.isfinite().all()
        x_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        self.assert_close(tail.grad, σ, atol=1e-3, rtol=1e-1)

    def test_negative_mu_matches_positive_mu(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = GAUSSIAN_TO_BIMODAL[name]
        inverse_impl = BIMODAL_TO_GAUSSIAN[name]
        y = self.make_full_range(dtype=dtype, device=device)
        x = self.make_full_range(dtype=dtype, device=device)
        μ_pos = torch.tensor(mean, dtype=dtype, device=device)
        μ_neg = torch.tensor(-mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)

        self.assert_close(
            forward_impl(y, μ_pos, σ),
            forward_impl(y, μ_neg, σ),
        )
        self.assert_close(
            inverse_impl(x, μ_pos, σ),
            inverse_impl(x, μ_neg, σ),
        )


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("stdv", BimodalTest.STDVS, ids="stdv={}".format)
@pytest.mark.parametrize("mean", BimodalTest.MEANS, ids="mean={}".format)
@pytest.mark.parametrize("name", GAUSSIAN_TO_BIMODAL_VALUE_AND_JAC, ids=str)
class TestGaussianToBimodalValueAndGrad(BimodalTest):
    def test_gradcheck(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL_VALUE_AND_JAC[name]
        y = self.make_safe_y_range(mean, stdv, dtype=dtype, device=device)
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)

        atol, rtol, eps = self.INVERSE_GRADCHECK_TOL[dtype]
        gradcheck(
            impl,
            (y, μ, σ),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )

    def test_reversible(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        inverse_impl = GAUSSIAN_TO_BIMODAL_VALUE_AND_JAC[name]
        forward_impl = bimodal_to_gaussian_value_and_jac_py
        y = self.make_safe_y_range(mean, stdv, dtype=dtype, device=device)

        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)

        x, d_y = inverse_impl(y, μ, σ)
        y_inv, d_x = forward_impl(x, μ, σ)
        y_inv.sum().backward()
        assert y.grad is not None

        atol, rtol = self.TOL[dtype]
        self.assert_close(y_inv, y, atol=atol, rtol=rtol)
        self.assert_close(d_x * d_y, 1.0, atol=atol, rtol=rtol)
        self.assert_close(y.grad, 1.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", MIXTURE_TO_GAUSSIAN, ids=str)
@pytest.mark.parametrize(("weights", "means", "stdvs"), TestMixture.CASES)
class TestMixtureToGaussian(TestMixture):
    def test_gradcheck(
        self,
        name: str,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = MIXTURE_TO_GAUSSIAN[name]
        x = self.make_full_range(means, stdvs, dtype=dtype, device=device)
        self.assert_gradcheck(
            impl, x, weights, means, stdvs, dtype=dtype, device=device
        )

    def test_reversible(
        self,
        name: str,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = MIXTURE_TO_GAUSSIAN[name]
        inverse_impl = GAUSSIAN_TO_MIXTURE[name]
        x = self.make_full_range(means, stdvs, dtype=dtype, device=device)

        omegas = torch.tensor(weights, dtype=dtype, device=device)
        mus = torch.tensor(means, dtype=dtype, device=device)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device)

        y = forward_impl(x, omegas, mus, sigmas)
        x_inv = inverse_impl(y, omegas, mus, sigmas)
        x_inv.sum().backward()
        assert x.grad is not None
        atol, rtol = self.TOL[dtype]
        self.assert_close(x_inv, x, atol=atol, rtol=rtol)
        self.assert_close(x.grad, 1.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(("weights", "means", "stdvs"), TestMixture.CASES)
@pytest.mark.parametrize("name", MIXTURE_TO_GAUSSIAN_VALUE_AND_JAC, ids=str)
class TestMixtureToGaussianValueAndGrad(TestMixture):
    def test_gradcheck(
        self,
        name: str,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = MIXTURE_TO_GAUSSIAN_VALUE_AND_JAC[name]
        x = self.make_full_range(means, stdvs, dtype=dtype, device=device)
        self.assert_gradcheck(
            impl, x, weights, means, stdvs, dtype=dtype, device=device
        )

    def test_reversible(
        self,
        name: str,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = MIXTURE_TO_GAUSSIAN_VALUE_AND_JAC[name]
        inverse_impl = gaussian_to_mixture_value_and_jac_py
        x = self.make_full_range(means, stdvs, dtype=dtype, device=device)

        omegas = torch.tensor(weights, dtype=dtype, device=device)
        mus = torch.tensor(means, dtype=dtype, device=device)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device)

        y, d_x = forward_impl(x, omegas, mus, sigmas)
        x_inv, d_y = inverse_impl(y, omegas, mus, sigmas)
        x_inv.sum().backward()
        assert x.grad is not None

        atol, rtol = self.TOL[dtype]
        self.assert_close(x_inv, x, atol=atol, rtol=rtol)
        self.assert_close(d_x * d_y, 1.0, atol=atol, rtol=rtol)
        self.assert_close(x.grad, 1.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(("weights", "means", "stdvs"), TestMixture.CASES)
@pytest.mark.parametrize("name", GAUSSIAN_TO_MIXTURE, ids=str)
class TestGaussianToMixture(TestMixture):
    def test_gradcheck(
        self,
        name: str,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_MIXTURE[name]
        y = self.make_full_range(means, stdvs, dtype=dtype, device=device)
        self.assert_gradcheck(
            impl, y, weights, means, stdvs, dtype=dtype, device=device
        )

    def test_reversible(
        self,
        name: str,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = GAUSSIAN_TO_MIXTURE[name]
        inverse_impl = MIXTURE_TO_GAUSSIAN[name]
        y = self.make_full_range(means, stdvs, dtype=dtype, device=device)

        omegas = torch.tensor(weights, dtype=dtype, device=device)
        mus = torch.tensor(means, dtype=dtype, device=device)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device)

        x = forward_impl(y, omegas, mus, sigmas)
        y_inv = inverse_impl(x, omegas, mus, sigmas)
        y_inv.sum().backward()
        assert y.grad is not None
        atol, rtol = self.TOL[dtype]
        self.assert_close(y_inv, y, atol=atol, rtol=rtol)
        self.assert_close(y.grad, 1.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(("weights", "means", "stdvs"), TestMixture.CASES)
@pytest.mark.parametrize("name", GAUSSIAN_TO_MIXTURE_VALUE_AND_JAC, ids=str)
class TestGaussianToMixtureValueAndGrad(TestMixture):
    def test_gradcheck(
        self,
        name: str,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_MIXTURE_VALUE_AND_JAC[name]
        y = self.make_full_range(means, stdvs, dtype=dtype, device=device)
        self.assert_gradcheck(
            impl, y, weights, means, stdvs, dtype=dtype, device=device
        )

    def test_reversible(
        self,
        name: str,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = GAUSSIAN_TO_MIXTURE_VALUE_AND_JAC[name]
        inverse_impl = mixture_to_gaussian_value_and_jac_py
        y = self.make_full_range(means, stdvs, dtype=dtype, device=device)

        omegas = torch.tensor(weights, dtype=dtype, device=device)
        mus = torch.tensor(means, dtype=dtype, device=device)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device)

        x, d_y = forward_impl(y, omegas, mus, sigmas)
        y_inv, d_x = inverse_impl(x, omegas, mus, sigmas)
        y_inv.sum().backward()
        assert y.grad is not None

        atol, rtol = self.TOL[dtype]
        self.assert_close(y_inv, y, atol=atol, rtol=rtol)
        self.assert_close(d_x * d_y, 1.0, atol=atol, rtol=rtol)
        self.assert_close(y.grad, 1.0, atol=atol, rtol=rtol)
