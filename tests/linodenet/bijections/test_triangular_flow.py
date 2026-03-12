import pytest
import torch

from linodenet.bijections import TriangularFlow
from tests.linodenet.bijections.fixtures import SEEDS


@pytest.mark.parametrize("seed", SEEDS, ids="seed={}".format)
@pytest.mark.parametrize("input_size", [4, 16, 64, 256], ids="input_size={}".format)
def test_invertibility(seed: int, input_size: int) -> None:
    r"""Check round trips and zero logabsdet for a unit lower-triangular flow."""
    torch.manual_seed(seed)
    value_atol = 1e-5
    value_rtol = 1e-5

    batch_size = 128
    flow = TriangularFlow(input_size)
    with torch.no_grad():
        flow.lower.copy_(0.1 * torch.randn_like(flow.lower).tril(diagonal=-1))

    x = torch.randn(batch_size, input_size)
    y, forward_logabsdet = flow.encode_and_logabsdet(x)
    xhat, inverse_logabsdet = flow.decode_and_logabsdet(y)

    assert y.shape == x.shape
    assert xhat.shape == x.shape
    assert forward_logabsdet.shape == (batch_size,)
    assert inverse_logabsdet.shape == (batch_size,)
    assert torch.allclose(xhat, x, atol=value_atol, rtol=value_rtol)
    assert torch.allclose(forward_logabsdet, torch.zeros_like(forward_logabsdet))
    assert torch.allclose(inverse_logabsdet, torch.zeros_like(inverse_logabsdet))

    y = torch.randn(batch_size, input_size)
    x, inverse_logabsdet = flow.decode_and_logabsdet(y)
    yhat, forward_logabsdet = flow.encode_and_logabsdet(x)

    assert torch.allclose(yhat, y, atol=value_atol, rtol=value_rtol)
    assert torch.allclose(forward_logabsdet, torch.zeros_like(forward_logabsdet))
    assert torch.allclose(inverse_logabsdet, torch.zeros_like(inverse_logabsdet))


def test_weight_is_unit_lower_triangular() -> None:
    r"""Check the constructed weight matrix has unit diagonal."""
    flow = TriangularFlow(8)
    with torch.no_grad():
        flow.lower.copy_(torch.randn_like(flow.lower))

    weight = flow.weight

    assert torch.allclose(weight.diag(), torch.ones(8, dtype=weight.dtype))
    assert torch.allclose(weight, weight.tril())
