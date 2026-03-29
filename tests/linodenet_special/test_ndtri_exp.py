import math

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.special import log_ndtr as scipy_log_ndtr, ndtri_exp as scipy_ndtri_exp_py
from torch import Tensor
from torch.autograd import gradcheck
from torch.special import log_ndtr

from linodenet_special.compiled import ndtri_exp as ndtri_exp_cpp
from linodenet_special.fallbacks.ndtri_exp import (
    _LOWER_CUTOFF,
    _UPPER_CUTOFF,
    ndtri_exp as ndtri_exp_py,
)
from tests.testing import DEVICES, DTYPES, TestCase

assert ndtri_exp_cpp is not None


def _scipy_ndtri_exp(values: Tensor, /) -> Tensor:
    reference = scipy_ndtri_exp_py(values.detach().cpu().numpy())
    tensor = torch.from_numpy(np.asarray(reference)).to(values.device, values.dtype)
    assert tensor.isfinite().all()
    return tensor


def _scipy_log_ndtr(values: Tensor, /) -> Tensor:
    reference = scipy_log_ndtr(values.detach().cpu().numpy())
    tensor = torch.from_numpy(np.asarray(reference)).to(values.device, values.dtype)
    assert tensor.isfinite().all()
    return tensor


IMPLS = {
    "py": ndtri_exp_py,
    "cpp": ndtri_exp_cpp,
}
TOL = {
    torch.float32: (1e-6, 1e-6),
    torch.float64: (1e-10, 1e-10),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_torch_log_ndtr_matches_scipy(dtype: torch.dtype) -> None:
    x = torch.linspace(-100, +100, steps=2048, dtype=dtype, requires_grad=True)
    actual = log_ndtr(x)
    reference = _scipy_log_ndtr(x)
    atol, rtol = TOL[dtype]
    assert torch.allclose(actual, reference, atol=atol, rtol=rtol)

    # check grads are finite.
    actual.sum().backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", IMPLS, ids=str)
class TestCorrectness(TestCase):
    N = 256
    RANGES = [
        (-80.0, _LOWER_CUTOFF - 1e-3),
        (_LOWER_CUTOFF, _UPPER_CUTOFF),
        (_UPPER_CUTOFF + 1e-6, -1e-6),
    ]

    TOL = {
        torch.float32: (1e-6, 1e-6),
        torch.float64: (1e-10, 1e-10),
    }

    REVERSIBLE_TOL = {
        torch.float32: (1e-5, 1e-5),
        torch.float64: (1e-10, 1e-10),
    }

    GRADCHECK_TOL = {}

    def test_special_values(self, name: str, dtype: torch.dtype, device: str) -> None:
        impl = IMPLS[name]

        # ndtri_exp_py(-∞) = ndtri(0) = -∞
        # ndtri_exp_py(0) = ndtri(1) = +∞
        # ndtri_exp_py(log(0.5)) = ndtri(0.5) = 0
        args = [-math.inf, math.log(0.5), 0.0]
        expected = [-math.inf, 0.0, math.inf]

        # check reference implementation
        np_dtype: type[np.floating] = (
            np.float32 if dtype is torch.float32 else np.float64
        )
        np_args = np.array(args, dtype=np_dtype)
        np_expected = np.array(expected, dtype=np_dtype)
        np_result = scipy_ndtri_exp_py(np_args)
        assert np.allclose(np_result, np_expected)

        # check our implementation
        pt_args = torch.tensor(args, dtype=dtype, device=device)
        pt_expected = torch.tensor(expected, dtype=dtype, device=device)
        pt_result = impl(pt_args)
        assert torch.allclose(pt_result, pt_expected)

    def test_domain(self, name: str, dtype: torch.dtype, device: str) -> None:
        impl = IMPLS[name]
        # ndtri_exp_py is defined for log_p <= 0
        # test on a geometric range of values from finfo.min to finfo.max
        # assert that for log_p > 0, the result is NaN
        # assert that for log_p <= 0, the result is finite or -inf
        finfo = torch.finfo(dtype)
        exp_min = math.log10(finfo.tiny)
        exp_max = math.log10(finfo.max)
        magnitudes = torch.logspace(
            exp_min, exp_max, steps=96, dtype=dtype, device=device
        )
        log_p_pos = magnitudes
        log_p_neg = -magnitudes
        log_p_zero = torch.tensor([0.0], dtype=dtype, device=device)

        result_pos = impl(log_p_pos)
        result_neg = impl(log_p_neg)
        result_zero = impl(log_p_zero)

        assert result_pos.isnan().all()
        assert (result_neg.isfinite() | result_neg.isneginf()).all()
        assert result_zero.isposinf().item()

    @pytest.mark.parametrize(
        ("lower", "upper"), RANGES, ids=["small", "medium", "large"]
    )
    def test_correctness(
        self,
        name: str,
        lower: float,
        upper: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        impl = IMPLS[name]
        log_p = torch.linspace(lower, upper, steps=self.N, dtype=dtype, device=device)
        expected = _scipy_ndtri_exp(log_p)
        actual = impl(log_p)
        atol, rtol = self.TOL[dtype]

        assert actual.isnan().eq(expected.isnan()).all()
        assert actual.isposinf().eq(expected.isposinf()).all()
        assert actual.isneginf().eq(expected.isneginf()).all()
        assert torch.allclose(actual, expected, atol=atol, rtol=rtol)

    def test_reversible(
        self,
        name: str,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        impl = IMPLS[name]

        x = torch.linspace(
            -8.0,
            +8.0,
            steps=self.N,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        y = log_ndtr(x)
        x_recovered = impl(y)
        x_recovered.sum().backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()

        atol, rtol = self.TOL[dtype]
        self.assert_close(x_recovered, x, atol=atol, rtol=rtol)
        self.assert_close(x.grad, 1.0, atol=atol, rtol=rtol)

    def test_gradcheck(self, name: str, dtype: torch.dtype, device: str) -> None:
        impl = IMPLS[name]
        log_p = torch.linspace(
            _LOWER_CUTOFF,
            _UPPER_CUTOFF,
            steps=100,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        if dtype is torch.float32:
            eps = 1e-4
            atol = 1e-3
            rtol = 1e-3
        else:
            eps = 1e-6
            atol = 1e-6
            rtol = 1e-6
        gradcheck(impl, (log_p,), eps=eps, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", IMPLS, ids=str)
class TestPerformance:
    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            (-80.0, _LOWER_CUTOFF - 1e-3),
            (_LOWER_CUTOFF, _UPPER_CUTOFF),
            (_UPPER_CUTOFF + 1e-6, -1e-6),
        ],
        ids=["small", "medium", "large"],
    )
    def test_performance(
        self,
        name: str,
        benchmark: BenchmarkFixture,
        lower: float,
        upper: float,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        impl = IMPLS[name]
        benchmark.group = f"ndtri_exp/{device}/{dtype}"
        log_p = torch.linspace(lower, upper, steps=128, dtype=dtype, device=device)

        def bench():
            torch.cuda.synchronize()
            impl(log_p)
            torch.cuda.synchronize()

        benchmark.pedantic(bench, (), iterations=10, rounds=20, warmup_rounds=20)
