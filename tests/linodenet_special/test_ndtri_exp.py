import math

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.special import ndtri_exp as scipy_ndtri_exp_py
from torch.autograd import gradcheck

from linodenet_special.compiled import ndtri_exp as ndtri_exp_cpp
from linodenet_special.fallbacks.ndtri_exp import (
    _LOWER_CUTOFF,
    _UPPER_CUTOFF,
    ndtri_exp as ndtri_exp_py,
)

from .fixtures import DEVICES, DTYPES

assert ndtri_exp_cpp is not None

ATOL = 1e-6
RTOL = 1e-3
IMPLS = {
    "py": ndtri_exp_py,
    "cpp": ndtri_exp_cpp,
}


def _scipy_reference(values: torch.Tensor) -> torch.Tensor:
    reference = scipy_ndtri_exp_py(values.detach().cpu().numpy())
    return torch.from_numpy(np.asarray(reference)).to(values.device, values.dtype)


def _assert_matches_reference(values: torch.Tensor, actual: torch.Tensor) -> None:
    reference = _scipy_reference(values)
    assert torch.isnan(actual).eq(torch.isnan(reference)).all()
    assert torch.isposinf(actual).eq(torch.isposinf(reference)).all()
    assert torch.isneginf(actual).eq(torch.isneginf(reference)).all()
    finite = torch.isfinite(reference)
    assert torch.allclose(actual[finite], reference[finite], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", IMPLS, ids=str)
def test_special_values(name: str, dtype: torch.dtype, device: str) -> None:
    impl = IMPLS[name]

    # ndtri_exp_py(-∞) = ndtri(0) = -∞
    # ndtri_exp_py(0) = ndtri(1) = +∞
    # ndtri_exp_py(log(0.5)) = ndtri(0.5) = 0
    args = [-math.inf, math.log(0.5), 0.0]
    expected = [-math.inf, 0.0, math.inf]

    # check reference implementation
    np_dtype: type[np.floating] = np.float32 if dtype is torch.float32 else np.float64
    np_args = np.array(args, dtype=np_dtype)
    np_expected = np.array(expected, dtype=np_dtype)
    np_result = scipy_ndtri_exp_py(np_args)
    assert np.allclose(np_result, np_expected)

    # check our implementation
    pt_args = torch.tensor(args, dtype=dtype, device=device)
    pt_expected = torch.tensor(expected, dtype=dtype, device=device)
    pt_result = impl(pt_args)
    assert torch.allclose(pt_result, pt_expected)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", IMPLS, ids=str)
def test_domain(name: str, dtype: torch.dtype, device: str) -> None:
    impl = IMPLS[name]
    # ndtri_exp_py is defined for log_p <= 0
    # test on a geometric range of values from finfo.min to finfo.max
    # assert that for log_p > 0, the result is NaN
    # assert that for log_p <= 0, the result is finite or -inf
    finfo = torch.finfo(dtype)
    exp_min = math.log10(finfo.tiny)
    exp_max = math.log10(finfo.max)
    magnitudes = torch.logspace(exp_min, exp_max, steps=96, dtype=dtype, device=device)
    log_p_pos = magnitudes
    log_p_neg = -magnitudes
    log_p_zero = torch.tensor([0.0], dtype=dtype, device=device)

    result_pos = impl(log_p_pos)
    result_neg = impl(log_p_neg)
    result_zero = impl(log_p_zero)

    assert result_pos.isnan().all()
    assert (result_neg.isfinite() | result_neg.isneginf()).all()
    assert result_zero.isposinf().item()


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-80.0, _LOWER_CUTOFF - 1e-3),
        (_LOWER_CUTOFF, _UPPER_CUTOFF),
        (_UPPER_CUTOFF + 1e-6, -1e-6),
    ],
    ids=["small", "medium", "large"],
)
@pytest.mark.parametrize("name", IMPLS, ids=str)
def test_correctness(
    name: str,
    lower: float,
    upper: float,
    device: str,
    dtype: torch.dtype,
) -> None:
    impl = IMPLS[name]
    log_p = torch.linspace(lower, upper, steps=256, dtype=dtype, device=device)
    _assert_matches_reference(log_p, impl(log_p))


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("name", IMPLS, ids=str)
def test_gradcheck(name: str, dtype: torch.dtype, device: str) -> None:
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
    name: str,
    benchmark: BenchmarkFixture,
    lower: float,
    upper: float,
    dtype: torch.dtype,
    device: str,
) -> None:
    impl = IMPLS[name]
    benchmark.group = f"ndtri_exp/{device}/{dtype}"
    log_p = torch.linspace(lower, upper, steps=256, dtype=dtype, device=device)

    def bench():
        torch.cuda.synchronize()
        impl(log_p)
        torch.cuda.synchronize()

    benchmark.pedantic(bench, (), iterations=10, rounds=20, warmup_rounds=20)
