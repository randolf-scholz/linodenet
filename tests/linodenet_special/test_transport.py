r"""Tests for transport maps."""

import pytest
import torch
from torch.autograd import gradcheck

from linodenet_special.fallbacks.transport import mixture_to_gaussian
from tests.linodenet_special.fixtures import DEVICES, DTYPES


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    ("weights", "means", "sigmas"),
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
        pytest.param(torch.randn(16, 8), id="batch"),
        pytest.param(torch.randn(()), id="scalar"),
        pytest.param([-3.0, -2.25, -1.5, -0.5, -0.1], id="p_branch"),
        pytest.param([0.1, 0.5, 1.5, 2.25, 3.0], id="q_branch"),
    ],
)
def test_mixture_to_gaussian_gradcheck(
    values: list[float],
    weights: list[float],
    means: list[float],
    sigmas: list[float],
    device: str,
    dtype: torch.dtype,
) -> None:
    x = torch.tensor(values, dtype=dtype, device=device, requires_grad=True)
    w = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
    mu = torch.tensor(means, dtype=dtype, device=device, requires_grad=True)
    sigma = torch.tensor(sigmas, dtype=dtype, device=device, requires_grad=True)

    if dtype is torch.float32:
        eps = 1e-4
        atol = 1e-2
        rtol = 1e-3
    else:
        eps = 1e-6
        atol = 1e-6
        rtol = 1e-6

    gradcheck(
        lambda z, ω, μ, σ: mixture_to_gaussian(z, ω / ω.sum(), μ, σ),
        (x, w, mu, sigma),
        eps=eps,
        atol=atol,
        rtol=rtol,
    )
