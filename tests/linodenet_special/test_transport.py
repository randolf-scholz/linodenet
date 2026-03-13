r"""Tests for transport maps."""

import pytest
import torch
from torch import Tensor
from torch.autograd import gradcheck

from linodenet_special.fallbacks.transport import mixture_to_gaussian
from tests.linodenet_special.fixtures import DEVICES, DTYPES


def _mixture_to_gaussian_with_raw_params(
    x: Tensor, logits: Tensor, means: Tensor, log_sigmas: Tensor
) -> Tensor:
    weights = logits.softmax(dim=-1)
    sigmas = log_sigmas.exp()
    return mixture_to_gaussian(x, weights, means, sigmas)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    ("logits", "means", "log_stds"),
    [
        pytest.param(
            [0.2, -0.4, 0.1],
            [-1.0, 0.5, 1.5],
            [-0.2, 0.1, -0.1],
            id="asymmetric",
        ),
        pytest.param(
            [-0.3, 0.4, -0.1],
            [-1.5, -0.5, 1.0],
            [0.0, -0.2, 0.2],
            id="shifted",
        ),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(torch.randn(16, 8), id="batch"),
        pytest.param(torch.randn(()), id="scalar"),
        pytest.param([-3.0, -2.25, -1.5, -0.5, -0.1], id="p_branch"),
        pytest.param([0.1, 0.5, 1.5, 2.25, 3.0], id="q_branch"),
    ],
)
def test_mixture_to_gaussian_gradcheck(
    values: list[float],
    logits: list[float],
    means: list[float],
    log_stds: list[float],
    device: str,
    dtype: torch.dtype,
) -> None:
    x = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
    log_w = torch.tensor(logits, dtype=dtype, device=device, requires_grad=True)
    mu = torch.tensor(means, dtype=dtype, device=device, requires_grad=True)
    log_sigma = torch.tensor(log_stds, dtype=dtype, device=device, requires_grad=True)

    if dtype is torch.float32:
        eps = 1e-4
        atol = 1e-2
        rtol = 1e-3
    else:
        eps = 1e-6
        atol = 1e-6
        rtol = 1e-6

    gradcheck(
        _mixture_to_gaussian_with_raw_params,
        (x, log_w, mu, log_sigma),
        eps=eps,
        atol=atol,
        rtol=rtol,
    )
