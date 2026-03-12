import torch

from linodenet.bijections import SplineFlow


def test_invertibility() -> None:
    torch.manual_seed(0)
    value_atol = 5e-3
    value_rtol = 1e-3
    logabsdet_atol = 2e-2
    logabsdet_rtol = 1e-3

    batch_size = 128
    num_heads = 4
    flow = SplineFlow(
        num_heads,
        num_flow_layers=3,
        num_bins=8,
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
    assert torch.allclose(xhat, x, atol=value_atol, rtol=value_rtol)
    assert torch.allclose(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=logabsdet_atol,
        rtol=logabsdet_rtol,
    )

    y = torch.randn(batch_size, num_heads)
    x, inverse_logabsdet = flow.decode_and_logabsdet(y)
    yhat, forward_logabsdet = flow.encode_and_logabsdet(x)

    assert x.shape == y.shape
    assert inverse_logabsdet.shape == (batch_size,)
    assert yhat.shape == y.shape
    assert forward_logabsdet.shape == (batch_size,)
    assert torch.allclose(yhat, y, atol=value_atol, rtol=value_rtol)
    assert torch.allclose(
        inverse_logabsdet + forward_logabsdet,
        torch.zeros_like(inverse_logabsdet),
        atol=logabsdet_atol,
        rtol=logabsdet_rtol,
    ), inverse_logabsdet + forward_logabsdet
