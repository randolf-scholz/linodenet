import pytest
import torch

from linodenet.bijections import SplineFlow
from tests.linodenet.bijections.fixtures import SEEDS


@pytest.mark.parametrize("seed", SEEDS, ids="seed={}".format)
@pytest.mark.parametrize("num_flow_layers", [1, 2, 3, 4], ids="layers={}".format)
@pytest.mark.parametrize("num_bins", [1, 2, 4, 8], ids="num_bins={}".format)
def test_invertibility(seed: int, num_flow_layers: int, num_bins: int) -> None:
    torch.manual_seed(seed)
    value_atol = 1.5e-2 * num_flow_layers
    value_rtol = 1e-3
    logabsdet_atol = 5e-3 * num_flow_layers
    logabsdet_rtol = 1e-3

    batch_size = 128
    num_heads = 4
    flow = SplineFlow(
        num_heads,
        num_flow_layers=num_flow_layers,
        num_bins=num_bins,
        x_bounds=(-3.0, 3.0),
        y_bounds=(-3.0, 3.0),
    )

    x = torch.randn(batch_size, num_heads)
    y, forward_logabsdet = flow.encode_and_logabsdet(x)
    xhat, inverse_logabsdet = flow.decode_and_logabsdet(y)

    assert y.shape == x.shape
    assert forward_logabsdet.shape == (batch_size,)
    assert xhat.shape == x.shape
    assert inverse_logabsdet.shape == (batch_size,)
    forward_inverse_abs_error = (xhat - x).abs()
    forward_inverse_rel_error = forward_inverse_abs_error / torch.maximum(
        torch.maximum(xhat.abs(), x.abs()),
        torch.full_like(xhat, torch.finfo(xhat.dtype).eps),
    )
    assert torch.allclose(xhat, x, atol=value_atol, rtol=value_rtol), (
        f"forward_inverse max_abs_error={forward_inverse_abs_error.max().item():.6e}, "
        f"max_rel_error={forward_inverse_rel_error.max().item():.6e}, "
        f"{value_atol=}, {value_rtol=}"
    )
    forward_inverse_logabsdet_error = (forward_logabsdet + inverse_logabsdet).abs()
    assert torch.allclose(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=logabsdet_atol,
        rtol=logabsdet_rtol,
    ), (
        f"forward_inverse_logabsdet max_abs_error="
        f"{forward_inverse_logabsdet_error.max().item():.6e}, "
        f"{logabsdet_atol=}, {logabsdet_rtol=}"
    )

    y = torch.randn(batch_size, num_heads)
    x, inverse_logabsdet = flow.decode_and_logabsdet(y)
    yhat, forward_logabsdet = flow.encode_and_logabsdet(x)

    assert x.shape == y.shape
    assert inverse_logabsdet.shape == (batch_size,)
    assert yhat.shape == y.shape
    assert forward_logabsdet.shape == (batch_size,)
    inverse_forward_abs_error = (yhat - y).abs()
    inverse_forward_rel_error = inverse_forward_abs_error / torch.maximum(
        torch.maximum(yhat.abs(), y.abs()),
        torch.full_like(yhat, torch.finfo(yhat.dtype).eps),
    )
    assert torch.allclose(yhat, y, atol=value_atol, rtol=value_rtol), (
        f"inverse_forward max_abs_error={inverse_forward_abs_error.max().item():.6e}, "
        f"max_rel_error={inverse_forward_rel_error.max().item():.6e}, "
        f"{value_atol=}, {value_rtol=}"
    )
    inverse_forward_logabsdet_error = (inverse_logabsdet + forward_logabsdet).abs()
    assert torch.allclose(
        inverse_logabsdet + forward_logabsdet,
        torch.zeros_like(inverse_logabsdet),
        atol=logabsdet_atol,
        rtol=logabsdet_rtol,
    ), (
        f"inverse_forward_logabsdet max_abs_error="
        f"{inverse_forward_logabsdet_error.max().item():.6e}, "
        f"{logabsdet_atol=}, {logabsdet_rtol=}"
    )
