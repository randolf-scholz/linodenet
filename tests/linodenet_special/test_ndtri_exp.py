import math

import numpy as np
import pytest
import torch
from scipy.special import ndtri_exp as scipy_ndtri_exp
from torch.autograd import gradcheck

from linodenet_special import ndtri_exp, ndtri_exp_naive
from linodenet_special.fallbacks.ndtri_exp import LOWER_CUTOFF, UPPER_CUTOFF

ATOL = 1e-6
RTOL = 1e-3


def _scipy_reference(values: torch.Tensor) -> torch.Tensor:
    reference = scipy_ndtri_exp(values.detach().cpu().numpy())
    return torch.from_numpy(np.asarray(reference)).to(values.device, values.dtype)


def _assert_matches_reference(values: torch.Tensor, actual: torch.Tensor) -> None:
    reference = _scipy_reference(values)
    assert torch.isnan(actual).eq(torch.isnan(reference)).all()
    assert torch.isposinf(actual).eq(torch.isposinf(reference)).all()
    assert torch.isneginf(actual).eq(torch.isneginf(reference)).all()
    finite = torch.isfinite(reference)
    assert torch.allclose(actual[finite], reference[finite], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids="dtype={}".format)
def test_ndtri_exp_special_values(dtype: str) -> None:
    # ndtri_exp(-∞) = ndtri(0) = -∞
    # ndtri_exp(0) = ndtri(1) = +∞
    # ndtri_exp(log(0.5)) = ndtri(0.5) = 0
    args = [-math.inf, math.log(0.5), 0.0]
    expected = [-math.inf, 0.0, math.inf]

    # check reference implementation
    np_dtype = np.float32 if dtype == "float32" else np.float64
    np_args = np.array(args, dtype=np_dtype)
    np_expected = np.array(expected, dtype=np_dtype)
    np_result = scipy_ndtri_exp(np_args)
    assert np.allclose(np_result, np_expected)

    # check our implementation
    pt_dtype = torch.float32 if dtype == "float32" else torch.float64
    pt_args = torch.tensor(args, dtype=pt_dtype)
    pt_expected = torch.tensor(expected, dtype=pt_dtype)
    pt_result = ndtri_exp(pt_args)
    assert torch.allclose(pt_result, pt_expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids="dtype={}".format)
def test_ndtri_exp_domain(dtype: str) -> None:
    # ndtri_exp is defined for log_p <= 0
    # test on a geometric range of values from finfo.min to finfo.max
    # assert that for log_p > 0, the result is NaN
    # assert that for log_p <= 0, the result is finite or -inf
    pt_dtype = torch.float32 if dtype == "float32" else torch.float64
    finfo = torch.finfo(pt_dtype)
    exp_min = math.log10(finfo.tiny)
    exp_max = math.log10(finfo.max)
    magnitudes = torch.logspace(exp_min, exp_max, steps=96, dtype=pt_dtype)
    log_p = torch.cat((-magnitudes, torch.zeros(1, dtype=pt_dtype), magnitudes))
    result = ndtri_exp(log_p)
    mask_pos = log_p > 0
    mask_neg = log_p < 0
    assert result[mask_pos].isnan().all()
    assert (result[mask_neg].isfinite() | result[mask_neg].isneginf()).all()


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-80.0, LOWER_CUTOFF - 1e-3),
        (LOWER_CUTOFF + 1e-6, UPPER_CUTOFF - 1e-6),
        (UPPER_CUTOFF + 1e-6, -1e-6),
    ],
    ids=["small", "mid", "large"],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids="dtype={}".format)
def test_ndtri_exp_correctness(lower: float, upper: float, dtype: str) -> None:
    pt_dtype = torch.float32 if dtype == "float32" else torch.float64
    log_p = torch.linspace(lower, upper, steps=256, dtype=pt_dtype)
    _assert_matches_reference(log_p, ndtri_exp(log_p))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-80.0, LOWER_CUTOFF - 1e-3),
        (LOWER_CUTOFF + 1e-6, UPPER_CUTOFF - 1e-6),
        (UPPER_CUTOFF + 1e-6, -1e-6),
    ],
    ids=["small", "mid", "large"],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"], ids="dtype={}".format)
def test_ndtri_exp_naive_correctness(lower: float, upper: float, dtype: str) -> None:
    pt_dtype = torch.float32 if dtype == "float32" else torch.float64
    log_p = torch.linspace(lower, upper, steps=256, dtype=pt_dtype)
    _assert_matches_reference(log_p, ndtri_exp_naive(log_p))


@pytest.mark.parametrize("dtype", ["float32", "float64"], ids="dtype={}".format)
def test_ndtri_exp_gradcheck(dtype: str) -> None:
    pt_dtype = torch.float32 if dtype == "float32" else torch.float64
    log_p = torch.linspace(-1.5, -0.3, steps=11, dtype=pt_dtype, requires_grad=True)
    gradcheck(ndtri_exp, (log_p,), eps=1e-6, atol=1e-4, rtol=1e-3)
