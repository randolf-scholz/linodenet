r"""Tests for ProFITi components."""

import torch
from torch import nn
from torch.func import grad, vmap
from torch.testing import assert_close

from linodenet.forecasting.grafiti import Grafiti
from linodenet.forecasting.profiti import (
    ConditionalFlowSequence,
    ProFITi,
    ProFITiBlock,
    ProFITiConfig,
    Shiesh,
    TriangularAttention,
)
from linodenet.forecasting.utils import BatchedTripletArgs


def test_shiesh_encode_decode_roundtrip_with_logabsdet() -> None:
    r"""Check that Shiesh encode/decode are inverse maps."""
    transform = Shiesh(t=1.0, a=1.0)
    x = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)

    y, forward_logabsdet = transform.encode_and_logabsdet(x)
    xhat, inverse_logabsdet = transform.decode_and_logabsdet(y)

    assert_close(xhat, x, atol=1e-12, rtol=1e-12)
    assert_close(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-12,
        rtol=1e-12,
    )


def test_shiesh_logabsdet_matches_autograd_derivative() -> None:
    r"""Check Shiesh logabsdet against autograd derivatives."""
    transform = Shiesh(t=1.0, a=1.0)
    x = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)

    _, actual = transform.encode_and_logabsdet(x)
    derivative = vmap(grad(lambda z: transform.encode_and_logabsdet(z)[0]))(x)
    expected = derivative.abs().log()

    assert actual.isfinite().all()
    assert expected.isfinite().all()
    assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_triangular_attention_encode_decode_roundtrip_with_logabsdet() -> None:
    r"""Check that triangular attention encode/decode are inverse maps."""
    torch.manual_seed(0)
    attention = TriangularAttention(dim_context=4, dim_hidden=3).to(dtype=torch.float64)
    context = torch.randn(5, 4, dtype=torch.float64)
    x = torch.randn(5, dtype=torch.float64)

    y, forward_logabsdet = attention.encode_and_logabsdet(x, context)
    xhat, inverse_logabsdet = attention.decode_and_logabsdet(y, context)

    assert_close(xhat, x, atol=1e-12, rtol=1e-12)
    assert_close(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-12,
        rtol=1e-12,
    )


def test_triangular_attention_logabsdet_matches_dense_jacobian() -> None:
    r"""Check triangular attention logabsdet against a dense Jacobian."""
    torch.manual_seed(0)
    num_steps = 5
    attention = TriangularAttention(dim_context=4, dim_hidden=3).to(dtype=torch.float64)
    context = torch.randn(num_steps, 4, dtype=torch.float64)
    x = torch.randn(num_steps, dtype=torch.float64)

    _, actual = attention.encode_and_logabsdet(x, context)

    def encode_flat(z: torch.Tensor, /) -> torch.Tensor:
        y, _ = attention.encode_and_logabsdet(z.reshape(num_steps), context)
        return y.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(encode_flat, x.reshape(-1))
    _, expected = torch.linalg.slogdet(jacobian)

    assert actual.isfinite()
    assert expected.isfinite()
    assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_profiti_block_encode_decode_roundtrip_with_logabsdet() -> None:
    r"""Check that a ProFITi block encode/decode are inverse maps."""
    torch.manual_seed(0)
    num_steps = 5
    latent_dim = 4
    block = ProFITiBlock(latent_dim=latent_dim, num_layers=1).to(dtype=torch.float64)
    context = torch.randn(num_steps, latent_dim, dtype=torch.float64)
    x = torch.randn(num_steps, dtype=torch.float64)

    y, forward_logabsdet = block.encode_and_logabsdet(x, context)
    xhat, inverse_logabsdet = block.decode_and_logabsdet(y, context)

    assert_close(xhat, x, atol=1e-10, rtol=1e-10)
    assert_close(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-10,
        rtol=1e-10,
    )


def test_profiti_block_logabsdet_matches_dense_jacobian() -> None:
    r"""Check a ProFITi block logabsdet against a dense Jacobian."""
    torch.manual_seed(0)
    num_steps = 5
    latent_dim = 4
    block = ProFITiBlock(latent_dim=latent_dim, num_layers=1).to(dtype=torch.float64)
    context = torch.randn(num_steps, latent_dim, dtype=torch.float64)
    x = torch.randn(num_steps, dtype=torch.float64)

    _, actual = block.encode_and_logabsdet(x, context)

    def encode_flat(z: torch.Tensor, /) -> torch.Tensor:
        y, _ = block.encode_and_logabsdet(z.reshape(num_steps), context)
        return y.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(encode_flat, x.reshape(-1))
    _, expected = torch.linalg.slogdet(jacobian)

    assert actual.isfinite()
    assert expected.isfinite()
    assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_flow_sequence_encode_decode_roundtrip_with_logabsdet() -> None:
    r"""Check that a sequence of ProFITi blocks is invertible."""
    torch.manual_seed(0)
    num_steps = 5
    latent_dim = 4
    flow = ConditionalFlowSequence(
        [
            ProFITiBlock(latent_dim=latent_dim, num_layers=1),
            ProFITiBlock(latent_dim=latent_dim, num_layers=1),
        ]
    ).to(dtype=torch.float64)
    context = torch.randn(num_steps, latent_dim, dtype=torch.float64)
    x = torch.randn(num_steps, dtype=torch.float64)

    y, forward_logabsdet = flow.encode_and_logabsdet(x, context)
    xhat, inverse_logabsdet = flow.decode_and_logabsdet(y, context)

    assert_close(xhat, x, atol=1e-10, rtol=1e-10)
    assert_close(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-10,
        rtol=1e-10,
    )


def test_flow_sequence_logabsdet_matches_dense_jacobian() -> None:
    r"""Check a sequence of ProFITi blocks against a dense Jacobian."""
    torch.manual_seed(0)
    num_steps = 5
    latent_dim = 4
    flow = ConditionalFlowSequence(
        [
            ProFITiBlock(latent_dim=latent_dim, num_layers=1),
            ProFITiBlock(latent_dim=latent_dim, num_layers=1),
        ]
    ).to(dtype=torch.float64)
    context = torch.randn(num_steps, latent_dim, dtype=torch.float64)
    x = torch.randn(num_steps, dtype=torch.float64)

    _, actual = flow.encode_and_logabsdet(x, context)

    def encode_flat(z: torch.Tensor, /) -> torch.Tensor:
        y, _ = flow.encode_and_logabsdet(z.reshape(num_steps), context)
        return y.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(encode_flat, x.reshape(-1))
    _, expected = torch.linalg.slogdet(jacobian)

    assert actual.isfinite()
    assert expected.isfinite()
    assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_grafiti_triplet_matches_combined_forward() -> None:
    r"""Check that sparse and combined GraFITi inputs produce the same embeddings."""
    torch.manual_seed(0)
    model = Grafiti(input_dim=3, hidden_dim=8, num_layers=2, num_heads=2)
    args = BatchedTripletArgs(
        context_times=torch.tensor(
            [
                [1.0, 1.0, 3.0, 4.0, 4.0],
                [0.0, 2.0, 2.0, torch.nan, torch.nan],
            ]
        ),
        context_channels=torch.tensor(
            [
                [0, 2, 1, 0, 2],
                [1, 0, 2, -1, -1],
            ]
        ),
        context_values=torch.tensor(
            [
                [10.0, 12.0, 31.0, 40.0, 42.0],
                [1.0, 20.0, 22.0, torch.nan, torch.nan],
            ]
        ),
        query_times=torch.tensor(
            [
                [2.0, 4.0, 4.0, torch.nan],
                [1.0, 3.0, 3.0, 5.0],
            ]
        ),
        query_channels=torch.tensor(
            [
                [0, 1, 2, -1],
                [0, 1, 2, 1],
            ]
        ),
        query_values=torch.tensor(
            [
                [200.0, 410.0, 420.0, torch.nan],
                [100.0, 310.0, 320.0, 510.0],
            ]
        ),
    )
    combined = args.to_combined()
    combined_values = combined.values.masked_fill(combined.query_mask, 0.0)

    expected = model.forward_combined(
        combined.times,
        combined_values,
        combined.context_mask,
        combined.query_mask,
    )
    actual = model.forward_triplet(
        args.context_times,
        args.context_channels,
        args.context_values,
        args.query_times,
        args.query_channels,
    )

    assert_close(actual, expected)


def test_profiti_from_config_uses_grafiti_and_flow_sequence() -> None:
    r"""Check that ProFITi.from_config wires the default submodules."""
    config = ProFITiConfig(
        input_dim=7,
        num_heads=2,
        latent_dim=12,
        num_layers=3,
    )

    model = ProFITi.from_config(config)

    assert isinstance(model.context_embedding, Grafiti)
    assert model.context_embedding.channel_init.in_features == config.input_dim
    assert model.context_embedding.hidden_dim == config.latent_dim
    assert model.context_embedding.num_heads == config.num_heads
    assert model.context_embedding.num_layers == config.num_layers
    assert isinstance(model.conditional_flow, ConditionalFlowSequence)
    assert not isinstance(model.conditional_flow, nn.Identity)
    assert len(model.conditional_flow) == config.num_layers
    assert all(isinstance(layer, ProFITiBlock) for layer in model.conditional_flow)
