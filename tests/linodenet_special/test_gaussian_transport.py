r"""Tests for transport maps."""

import math

import pytest
import torch
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
    bimodal_to_gaussian_value_and_jac as bimodal_to_gaussian_value_and_jac_py,
    gaussian_to_bimodal as gaussian_to_bimodal_py,
    gaussian_to_bimodal_value_and_jac as gaussian_to_bimodal_value_and_jac_py,
    gaussian_to_mixture as gaussian_to_mixture_py,
    mixture_to_gaussian as mixture_to_gaussian_py,
    mixture_to_gaussian_value_and_jac as mixture_to_gaussian_value_and_jac_py,
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


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", BIMODAL_TO_GAUSSIAN, ids=str)
class TestBimodalToGaussian(TestCase):
    SEED = 0
    X_MIN = -20
    X_MAX = 20
    N = 256

    STDVS = [0.25, 0.5, 1, 2, 10]
    MEANS = [0.1, 0.5, 1, 2, 4]

    TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-7, 1e-7),
    }

    GRADCHECK_TOL = {
        torch.float32: (1e-3, 1e-3, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    @staticmethod
    def get_x_star(mean: float, stdv: float) -> float:
        """Critical point of the piecewise-linear approximation.

        Given λ=Ψ'(0)=exp(-½μ²/σ²)/σ, it's λx = (x±μ)/σ ⟺ x = ±μ/(1-λσ)
        """
        lam = math.exp(-0.5 * (mean / stdv) ** 2) / stdv
        return mean * min(1.0, abs(1 / (1 - lam * stdv)))

    @classmethod
    def make_test_range(
        cls, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        r"""Construct a numerically useful test range inside $[-μ, μ]$.

        The slope at the origin is $Ψ'(0)=σ⁻¹ℯ^{-½(μ/σ)²}$. For a given floating
        point dtype we treat the central region as numerically flat once the
        local slope drops below $√ρ/σ$, where $ρ$ is the decimal resolution.
        Solving the corresponding Gaussian tail model yields the exclusion radius

        .. math:: x_\text{safe} = \max(0, μ - σ\sqrt{-\log ρ}).

        If $μ/σ$ is smaller than the dtype-dependent threshold $√{-\log ρ}$, we
        keep the full interval $[-μ, μ]$. Otherwise, we exclude the flat center
        and use $[-μ, -x_\text{safe}] ∪ [x_\text{safe}, μ]$.
        """
        # sqrt(-log(resolution)) is about 3.7 for float32 and 5.9 for float64.
        threshold = math.sqrt(-math.log(torch.finfo(dtype).resolution)) - 0.5
        if mean / stdv <= threshold:
            return torch.linspace(-mean, mean, steps=cls.N, dtype=dtype, device=device)
        x_safe = max(0.0, mean - stdv * threshold)
        x_neg = torch.linspace(
            -mean,
            -x_safe,
            steps=cls.N // 2,
            dtype=dtype,
            device=device,
        )
        x_pos = torch.linspace(
            x_safe,
            mean,
            steps=cls.N - cls.N // 2,
            dtype=dtype,
            device=device,
        )
        return torch.cat([x_neg, x_pos])

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_bimodal_to_gaussian_forward(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = (torch.exp(-0.5 * (μ / σ) ** 2) / σ).item()

        zero = torch.tensor(0, dtype=dtype, device=device)
        y_zero = impl(zero, μ, σ)
        self.assert_close(y_zero, zero)

        x1 = torch.linspace(0, self.X_MAX, steps=self.N, dtype=dtype, device=device)
        y1 = impl(x1, μ, σ)
        assert y1.dtype == dtype
        assert y1.isfinite().all()

        x2 = torch.linspace(0, self.X_MIN, steps=self.N, dtype=dtype, device=device)
        y2 = impl(x2, μ, σ)
        assert y2.dtype == dtype
        assert y2.isfinite().all()

        self.assert_close(y1, -y2)

        x_tail = max(100.0, μ.item() * max(1, 1 / (1 - λ)))
        assert x_tail > 0
        x1 = torch.linspace(
            100 * x_tail, 1000 * x_tail, steps=self.N, dtype=dtype, device=device
        )
        x2 = -x1
        tail1 = (x1 - torch.sign(x1) * μ) / σ
        tail2 = (x2 - torch.sign(x2) * μ) / σ
        y1 = impl(x1, μ, σ)
        y2 = impl(x2, μ, σ)
        assert y1.isfinite().all()
        assert y2.isfinite().all()
        self.assert_close(y1, tail1)
        self.assert_close(y2, tail2)

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_bimodal_to_gaussian_backward(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / stdv
        g_rtol = 2**-4
        lower_grad_bound = max(0, λ.item() * (1 - g_rtol))
        upper_grad_bound = 1 / stdv

        x1 = torch.linspace(
            0,
            self.X_MAX,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y1 = impl(x1, μ, σ)
        y1.sum().backward()
        assert x1.grad is not None
        assert x1.grad.isfinite().all()
        assert x1.grad.max() <= upper_grad_bound
        assert x1.grad.min() >= lower_grad_bound
        self.assert_close(x1.grad[0], λ, rtol=g_rtol)

        x2 = torch.linspace(
            0,
            self.X_MIN,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y2 = impl(x2, μ, σ)
        y2.sum().backward()
        assert x2.grad is not None
        assert x2.grad.isfinite().all()
        assert x2.grad.max() <= upper_grad_bound
        assert x2.grad.min() >= lower_grad_bound
        self.assert_close(x2.grad[0], λ, rtol=g_rtol)
        self.assert_close(x1.grad, x2.grad)

        x_tail = self.get_x_star(mean, stdv)
        assert x_tail > 0
        tail_values = torch.linspace(
            10 * x_tail, 100 * x_tail, steps=self.N, dtype=dtype, device=device
        )
        tail = torch.cat([tail_values, tail_values.neg()]).requires_grad_()
        y_tail = impl(tail, μ, σ)
        assert y_tail.isfinite().all()
        y_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        self.assert_close(tail.grad, upper_grad_bound, rtol=0.5)

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_hard_contract_approximation(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        r"""When the gaussians are well separated, we can approximate with hard_bend."""
        impl = BIMODAL_TO_GAUSSIAN[name]
        x = self.make_test_range(mean, stdv, dtype=dtype, device=device)
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = torch.exp(-0.5 * (μ / σ) ** 2) / σ

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
        self.assert_upper_bounded(y - y_approx, μ / σ)

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_bimodal_to_gaussian_gradcheck(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        x_narrow = self.make_test_range(mean, stdv, dtype, device).requires_grad_()

        atol, rtol, eps = self.GRADCHECK_TOL[dtype]
        gradcheck(
            impl,
            (x_narrow, μ, σ),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )

    @pytest.mark.parametrize("stdv", [0.5, 1, 2, 10], ids="stdv={}".format)
    @pytest.mark.parametrize("mean", [0.1, 0.5, 1, 2], ids="mean={}".format)
    def test_reversible(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = BIMODAL_TO_GAUSSIAN[name]
        inverse_impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        x = self.make_test_range(mean, stdv, dtype, device).requires_grad_()
        y = forward_impl(x, μ, σ)
        x_inv = inverse_impl(y, μ, σ)
        z = x_inv.sum()
        z.backward()
        assert x.grad is not None
        atol, rtol = self.TOL[dtype]
        self.assert_close(x_inv, x, atol=atol, rtol=rtol)
        self.assert_close(x.grad, 1.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", BIMODAL_TO_GAUSSIAN_VALUE_AND_JAC, ids=str)
class TestBimodalToGaussianValueAndJac(TestCase):
    SEED = 0
    N = 256

    STDVS = [0.25, 0.5, 1, 2, 10]
    MEANS = [0.1, 0.5, 1, 2, 4]

    GRADCHECK_TOL = {
        torch.float32: (1e-3, 1e-3, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    @staticmethod
    def get_x_star(mean: float, stdv: float) -> float:
        """Critical point of the piecewise-linear approximation.

        Given λ=Ψ'(0)=exp(-½μ²/σ²)/σ, it's λx = (x±μ)/σ ⟺ x = ±μ/(1-λσ)
        """
        lam = math.exp(-0.5 * (mean / stdv) ** 2) / stdv
        return mean * min(1.0, abs(1 / (1 - lam * stdv)))

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_gradcheck(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = BIMODAL_TO_GAUSSIAN_VALUE_AND_JAC[name]
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        x_neg = torch.linspace(
            -mean - 3 * stdv,
            -mean + 3 * stdv,
            steps=self.N // 2,
            dtype=dtype,
            device=device,
        )
        x_pos = torch.linspace(
            mean - 3 * stdv,
            mean + 3 * stdv,
            steps=self.N // 2,
            dtype=dtype,
            device=device,
        )
        x_narrow = torch.cat([x_neg, x_pos]).requires_grad_()

        atol, rtol, eps = self.GRADCHECK_TOL[dtype]
        gradcheck(
            impl,
            (x_narrow, μ, σ),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", GAUSSIAN_TO_BIMODAL, ids=str)
class TestGaussianToBimodal(TestCase):
    SEED = 0
    X_MIN = -20
    X_MAX = 20
    N = 256
    STDVS = [1, 2, 3]
    MEANS = [0.5, 1, 2]

    TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-7, 1e-7),
    }

    GRADCHECK_TOL = {
        torch.float32: (1e-2, 1e-2, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    @staticmethod
    def get_x_star(mean: float, stdv: float) -> float:
        """Critical point of the piecewise-linear approximation.

        Given λ=Ψ⁻¹'(0)=σ⋅exp(½μ²/σ²), it's λx = σx±μ ⟺ x = ±μ/(λ-σ),
        """
        lam = stdv * math.exp(0.5 * (mean / stdv) ** 2)
        return abs(mean / (lam - stdv))

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_hard_expand_approximation(
        self, name: str, mean, stdv, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL[name]
        y = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N, dtype=dtype, device=device
        )
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = (torch.exp(-0.5 * (μ / σ) ** 2) / σ).item()

        x = impl(y, μ, σ)
        assert x.dtype == dtype
        assert x.isfinite().all(), (
            "gaussian_to_bimodal should produce finite outputs for finite inputs"
        )

        x_approx = hard_bend(y, 1 / λ, μ, σ)
        assert x_approx.dtype == dtype
        assert x_approx.isfinite().all(), (
            "Hard-expand approximation should produce finite outputs"
        )
        self.assert_upper_bounded(x - x_approx, μ * σ, atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_gaussian_to_bimodal_forward(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ = (torch.exp(-0.5 * (μ / σ) ** 2) / σ).item()

        zero = torch.tensor(0, dtype=dtype, device=device)
        x_zero = impl(zero, μ, σ)
        self.assert_close(x_zero, zero)

        y1 = torch.linspace(0, self.X_MAX, steps=self.N, dtype=dtype, device=device)
        x1 = impl(y1, μ, σ)
        assert x1.dtype == dtype
        assert x1.isfinite().all()

        y2 = torch.linspace(0, self.X_MIN, steps=self.N, dtype=dtype, device=device)
        x2 = impl(y2, μ, σ)
        assert x2.dtype == dtype
        assert x2.isfinite().all()

        self.assert_close(x1, -x2)

        y_tail = max(100.0, μ.abs().item() / (λ - 1))
        assert y_tail > 0
        y1 = torch.linspace(
            100 * y_tail, 1000 * y_tail, steps=self.N, dtype=dtype, device=device
        )
        y2 = -y1
        tail1 = σ * y1 - μ
        tail2 = σ * y2 + μ
        x1 = impl(y1, μ, σ)
        x2 = impl(y2, μ, σ)
        assert x1.isfinite().all()
        assert x2.isfinite().all()
        self.assert_close(x1, tail1, rtol=1e-3)
        self.assert_close(x2, tail2, rtol=1e-3)

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_gaussian_to_bimodal_backward(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        λ_log = 0.5 * (μ / σ) ** 2 + σ.log()
        g_rtol = 2**-4
        log_tol = math.log(1 + g_rtol)
        log_grad_bound = λ_log + log_tol

        y1 = torch.linspace(
            0,
            self.X_MAX,
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
            0,
            self.X_MIN,
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

        y_tail = self.get_x_star(mean, stdv)
        assert y_tail > 0
        tail_values = torch.linspace(
            10 * y_tail, 100 * y_tail, steps=self.N, dtype=dtype, device=device
        )
        tail = torch.cat([tail_values, tail_values.neg()]).requires_grad_()
        x_tail = impl(tail, μ, σ)
        assert x_tail.isfinite().all()
        x_tail.sum().backward()
        assert tail.grad is not None
        assert tail.grad.isfinite().all()
        # FIXME: huge rtol needed?!
        self.assert_close(tail.grad, σ, atol=1e-3, rtol=1e-1)

    @pytest.mark.parametrize("stdv", [0.5, 1, 2, 10], ids="stdv={}".format)
    @pytest.mark.parametrize("mean", [0.1, 0.5, 1, 2], ids="mean={}".format)
    def test_reversible(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        inverse_impl = GAUSSIAN_TO_BIMODAL[name]
        forward_impl = BIMODAL_TO_GAUSSIAN[name]
        μ = torch.tensor(mean, dtype=dtype, device=device)
        σ = torch.tensor(stdv, dtype=dtype, device=device)
        y = torch.linspace(
            self.X_MIN,
            self.X_MAX,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        x_inv = inverse_impl(y, μ, σ)
        y_inv = forward_impl(x_inv, μ, σ)
        z = y_inv.sum()
        z.backward()
        assert y.grad is not None
        atol, rtol = self.TOL[dtype]
        self.assert_close(y_inv, y, atol=atol, rtol=rtol)
        self.assert_close(y.grad, 1.0, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_gaussian_to_bimodal_gradcheck(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL[name]
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        y_star = self.get_x_star(mean, stdv)
        y_narrow = torch.linspace(
            -y_star / 2,
            y_star / 2,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        atol, rtol, eps = self.GRADCHECK_TOL[dtype]
        gradcheck(
            impl,
            (y_narrow, μ, σ),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_negative_mu_matches_positive_mu(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        forward_impl = GAUSSIAN_TO_BIMODAL[name]
        inverse_impl = BIMODAL_TO_GAUSSIAN[name]
        y = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N, dtype=dtype, device=device
        )
        x = torch.linspace(
            self.X_MIN, self.X_MAX, steps=self.N, dtype=dtype, device=device
        )
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
@pytest.mark.parametrize("name", GAUSSIAN_TO_BIMODAL_VALUE_AND_JAC, ids=str)
class TestGaussianToBimodalValueAndJac(TestCase):
    SEED = 0
    N = 256
    STDVS = [1, 2, 3]
    MEANS = [0.5, 1, 2]

    GRADCHECK_TOL = {
        torch.float32: (1e-2, 1e-2, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    @pytest.mark.parametrize("stdv", STDVS, ids="stdv={}".format)
    @pytest.mark.parametrize("mean", MEANS, ids="mean={}".format)
    def test_gradcheck(
        self, name: str, mean: float, stdv: float, dtype: torch.dtype, device: str
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_BIMODAL_VALUE_AND_JAC[name]
        μ = torch.tensor(mean, dtype=dtype, device=device, requires_grad=True)
        σ = torch.tensor(stdv, dtype=dtype, device=device, requires_grad=True)
        x_neg = torch.linspace(
            -mean - 3 * stdv,
            -mean + 3 * stdv,
            steps=self.N // 2,
            dtype=dtype,
            device=device,
        )
        x_pos = torch.linspace(
            mean - 3 * stdv,
            mean + 3 * stdv,
            steps=self.N // 2,
            dtype=dtype,
            device=device,
        )
        x_narrow = torch.cat([x_neg, x_pos])
        y_narrow = bimodal_to_gaussian_py(
            x_narrow, μ.detach(), σ.detach()
        ).requires_grad_()

        atol, rtol, eps = self.GRADCHECK_TOL[dtype]
        gradcheck(
            impl,
            (y_narrow, μ, σ),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", MIXTURE_TO_GAUSSIAN, ids=str)
@pytest.mark.parametrize(
    ("weights", "means", "stdvs"),
    [
        pytest.param(
            [0.4, 0.25, 0.35],
            [-1.0, 0.5, 1.5],
            [0.8, 1.1, 0.9],
            id="asymmetric",
        ),
        pytest.param(
            [0.2, 0.5, 0.3],
            [-1.5, -0.5, 1.0],
            [1.0, 0.8, 1.2],
            id="shifted",
        ),
    ],
)
class TestMixtureToGaussian(TestCase):
    SEED = 0
    N = 256

    TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-7, 1e-7),
    }

    GRADCHECK_TOL = {
        torch.float32: (1e-3, 1e-3, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

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
        omegas = torch.tensor(weights, dtype=dtype, device=device)
        mus = torch.tensor(means, dtype=dtype, device=device)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device)
        x_min = torch.min(mus - 3 * sigmas).item()
        x_max = torch.max(mus + 3 * sigmas).item()
        x = torch.linspace(
            x_min,
            x_max,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        y = forward_impl(x, omegas, mus, sigmas)
        x_inv = inverse_impl(y, omegas, mus, sigmas)
        x_inv.sum().backward()
        assert x.grad is not None
        atol, rtol = self.TOL[dtype]
        self.assert_close(x_inv, x, atol=atol, rtol=rtol)
        self.assert_close(x.grad, 1.0, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([[-1.25, -0.5], [0.25, 1.75]], id="batch"),
            pytest.param(0.375, id="scalar"),
            pytest.param([-3.0, -2.25, -1.5, -0.5, -0.1], id="p_branch"),
            pytest.param([0.1, 0.5, 1.5, 2.25, 3.0], id="q_branch"),
        ],
    )
    def test_gradcheck(
        self,
        name: str,
        values: list[float] | float,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = MIXTURE_TO_GAUSSIAN[name]
        x = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
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
@pytest.mark.parametrize("name", GAUSSIAN_TO_MIXTURE, ids=str)
class TestGaussianToMixture(TestCase):
    SEED = 0
    N = 256

    TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-7, 1e-7),
    }

    GRADCHECK_TOL = {
        torch.float32: (1e-2, 1e-2, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    @pytest.mark.parametrize(
        ("weights", "means", "stdvs"),
        [
            pytest.param(
                [0.4, 0.25, 0.35],
                [-1.0, 0.5, 1.5],
                [0.8, 1.1, 0.9],
                id="asymmetric",
            ),
            pytest.param(
                [0.2, 0.5, 0.3],
                [-1.5, -0.5, 1.0],
                [1.0, 0.8, 1.2],
                id="shifted",
            ),
        ],
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
        omegas = torch.tensor(weights, dtype=dtype, device=device)
        mus = torch.tensor(means, dtype=dtype, device=device)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device)
        y = torch.linspace(
            -4,
            4,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        x = forward_impl(y, omegas, mus, sigmas)
        y_inv = inverse_impl(x, omegas, mus, sigmas)
        y_inv.sum().backward()
        assert y.grad is not None
        atol, rtol = self.TOL[dtype]
        self.assert_close(y_inv, y, atol=atol, rtol=rtol)
        self.assert_close(y.grad, 1.0, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("weights", "means", "stdvs"),
        [
            pytest.param(
                [0.4, 0.25, 0.35],
                [-1.0, 0.5, 1.5],
                [0.8, 1.1, 0.9],
                id="asymmetric",
            ),
            pytest.param(
                [0.2, 0.5, 0.3],
                [-1.5, -0.5, 1.0],
                [1.0, 0.8, 1.2],
                id="shifted",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([[-2.0, -0.75], [0.25, 1.5]], id="batch"),
            pytest.param(-0.375, id="scalar"),
            pytest.param([-3.0, -1.5, -0.5, 0.0, 0.5], id="left"),
            pytest.param([0.0, 0.5, 1.5, 2.25, 3.0], id="right"),
        ],
    )
    def test_gradcheck(
        self,
        name: str,
        values: list[float] | float,
        weights: list[float],
        means: list[float],
        stdvs: list[float],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        impl = GAUSSIAN_TO_MIXTURE[name]
        y = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
        omegas = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
        mus = torch.tensor(means, dtype=dtype, device=device, requires_grad=True)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device, requires_grad=True)

        atol, rtol, eps = self.GRADCHECK_TOL[dtype]
        gradcheck(
            lambda z, ω, μ, σ: impl(z, ω / ω.sum(), μ, σ),
            (y, omegas, mus, sigmas),
            eps=eps,
            atol=atol,
            rtol=rtol,
            fast_mode=True,
        )


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", MIXTURE_TO_GAUSSIAN_VALUE_AND_JAC, ids=str)
class TestMixtureToGaussianValueAndJac(TestCase):
    SEED = 0
    N = 256
    GRADCHECK_TOL = {
        torch.float32: (1e-3, 1e-3, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    @pytest.mark.parametrize(
        ("weights", "means", "stdvs"),
        [
            pytest.param(
                [0.4, 0.25, 0.35], [-1.0, 0.5, 1.5], [0.8, 1.1, 0.9], id="asymmetric"
            ),
            pytest.param(
                [0.2, 0.5, 0.3], [-1.5, -0.5, 1.0], [1.0, 0.8, 1.2], id="shifted"
            ),
        ],
    )
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
        omegas = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
        mus = torch.tensor(means, dtype=dtype, device=device, requires_grad=True)
        sigmas = torch.tensor(stdvs, dtype=dtype, device=device, requires_grad=True)

        mu_min = mus.min()
        mu_max = mus.max()
        sigma_max = sigmas.abs().max()

        x = torch.linspace(
            mu_min - sigma_max,
            mu_max + sigma_max,
            steps=self.N,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        atol, rtol, eps = self.GRADCHECK_TOL[dtype]
        gradcheck(
            lambda z, ω, μ, σ: impl(z, ω / ω.sum(), μ, σ),
            (x, omegas, mus, sigmas),
            atol=atol,
            rtol=rtol,
            eps=eps,
            fast_mode=True,
        )
