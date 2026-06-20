r"""Tests for ProFITi components."""

import math

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


class _TargetContext(nn.Module):
    r"""Minimal context encoder for ProFITi wrapper tests."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        _time_points: torch.Tensor,
        context_values: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        *batch_shape, _, _ = target_mask.shape
        max_targets = int(target_mask.sum(dim=(-2, -1)).max().item())
        return context_values.new_zeros(*batch_shape, max_targets, self.hidden_dim)


class _ScaleDecodeFlow(nn.Module):
    r"""Affine conditional flow with a constant decode scale."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def encode_and_logabsdet(
        self,
        x: torch.Tensor,
        _context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logabsdet = x.new_full(x.shape[:-1], -x.shape[-1] * math.log(self.scale))
        return x / self.scale, logabsdet

    def decode_and_logabsdet(
        self,
        y: torch.Tensor,
        _context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logabsdet = y.new_full(y.shape[:-1], y.shape[-1] * math.log(self.scale))
        return self.scale * y, logabsdet


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
    assert model.context_embedding.latent_dim == config.latent_dim
    assert model.context_embedding.num_heads == config.num_heads
    assert model.context_embedding.num_layers == config.num_layers
    assert isinstance(model.conditional_flow, ConditionalFlowSequence)
    assert not isinstance(model.conditional_flow, nn.Identity)
    assert len(model.conditional_flow) == config.num_layers
    assert all(isinstance(layer, ProFITiBlock) for layer in model.conditional_flow)


def test_profiti_sample_and_log_prob_uses_standard_normal_latent() -> None:
    r"""Check ProFITi samples via the flow and returns transformed log-density."""
    torch.manual_seed(0)
    scale = 2.0
    model = ProFITi(
        context_embedding=_TargetContext(hidden_dim=3),
        conditional_flow=_ScaleDecodeFlow(scale=scale),
    )
    context_times = torch.tensor([0.0, 1.0])
    context_values = torch.tensor([[1.0, torch.nan], [2.0, 3.0]])
    query_times = torch.tensor([2.0, 3.0, 4.0])
    query_mask = torch.tensor(
        [
            [True, False],
            [True, True],
            [False, True],
        ]
    )

    samples, log_prob = model.sample_and_log_prob(
        5,
        context_times=context_times,
        context_values=context_values,
        query_times=query_times,
        query_mask=query_mask,
    )

    assert samples.shape == (5, *query_mask.shape)
    assert log_prob.shape == (5,)
    assert samples[:, query_mask].isfinite().all()
    assert samples[:, ~query_mask].isnan().all()

    latents = samples[:, query_mask] / scale
    expected = -0.5 * (latents.square() + math.log(2.0 * math.pi)).sum(dim=-1)
    expected = expected - query_mask.sum() * math.log(scale)
    assert_close(log_prob, expected)


def test_profiti_sample_and_log_prob_handles_batch_dimensions() -> None:
    r"""Check ProFITi scatters flattened flow samples through batch dimensions."""
    torch.manual_seed(0)
    scale = 2.0
    model = ProFITi(
        context_embedding=_TargetContext(hidden_dim=3),
        conditional_flow=_ScaleDecodeFlow(scale=scale),
    )
    context_times = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    context_values = torch.tensor(
        [
            [[1.0, torch.nan], [2.0, 3.0]],
            [[4.0, 5.0], [torch.nan, 6.0]],
        ]
    )
    query_times = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    query_mask = torch.tensor(
        [
            [[True, False], [True, True], [False, True]],
            [[False, True], [True, False], [True, True]],
        ]
    )

    samples, log_prob = model.sample_and_log_prob(
        5,
        context_times=context_times,
        context_values=context_values,
        query_times=query_times,
        query_mask=query_mask,
    )

    assert samples.shape == (5, *query_mask.shape)
    assert log_prob.shape == (5, 2)
    assert samples[:, query_mask].isfinite().all()
    assert samples[:, ~query_mask].isnan().all()

    latents = torch.stack([samples[:, k, query_mask[k]] / scale for k in range(2)])
    expected = -0.5 * (latents.square() + math.log(2.0 * math.pi)).sum(dim=-1)
    expected = expected - query_mask[0].sum() * math.log(scale)
    assert_close(log_prob, expected.mT)
