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
