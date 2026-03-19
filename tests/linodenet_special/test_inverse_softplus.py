import math

import pytest
import torch
from torch.autograd import gradcheck
from torch.nn.functional import softplus

from linodenet_special.fallbacks import inverse_softplus
from tests.testing import DEVICES, DTYPES


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_inverse_softplus_special_values(dtype: torch.dtype, device: str) -> None:
    values = torch.tensor([0.0, 1.0, math.inf, -1.0], dtype=dtype, device=device)
    result = inverse_softplus(values)

    assert result[0].isneginf().item()
    assert result[1].isfinite().item()
    assert result[2].isposinf().item()
    assert result[3].isnan().item()


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_inverse_softplus_roundtrip_from_preimage(
    dtype: torch.dtype, device: str
) -> None:
    match dtype:
        case torch.float32:
            lower = -12.0
            upper = 12.0
            atol = rtol = 1e-5
        case torch.float64:
            lower = -20.0
            upper = 20.0
            atol = rtol = 1e-10
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")

    x = torch.linspace(
        lower,
        upper,
        steps=257,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    x_recovered = inverse_softplus(softplus(x))
    x_recovered.sum().backward()

    assert x.grad is not None
    torch.testing.assert_close(x_recovered, x, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, torch.ones_like(x), atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_inverse_softplus_roundtrip_from_image(dtype: torch.dtype, device: str) -> None:
    match dtype:
        case torch.float32:
            lower = math.log(torch.finfo(dtype).tiny)
            upper = 20.0
            atol = rtol = 1e-6
        case torch.float64:
            lower = math.log(torch.finfo(dtype).tiny)
            upper = 40.0
            atol = rtol = 1e-12
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")

    values = torch.logspace(
        lower / math.log(10),
        upper / math.log(10),
        steps=257,
        dtype=dtype,
        device=device,
    )
    recovered = softplus(inverse_softplus(values))

    torch.testing.assert_close(recovered, values, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_inverse_softplus_gradcheck(dtype: torch.dtype, device: str) -> None:
    if dtype is torch.float32:
        pytest.skip("gradcheck is only reliable in double precision")

    values = torch.logspace(
        -2,
        1,
        steps=64,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    eps = 1e-6
    atol = rtol = 1e-6

    gradcheck(inverse_softplus, (values,), eps=eps, atol=atol, rtol=rtol)
