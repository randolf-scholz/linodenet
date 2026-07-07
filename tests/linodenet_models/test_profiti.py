r"""Tests for ProFITi components."""

import math
from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import nn
from torch.func import grad, vmap
from torch.testing import assert_close

from linodenet_models.grafiti import Grafiti
from linodenet_models.profiti import (
    ConditionalFlowSequence,
    ProFITi,
    ProFITiBlock,
    ProFITiConfig,
    Shiesh,
    TriangularAttention,
)
from linodenet_models.utils import SplitTimeData

from .base import TestForecastingModel


class ProFITiTestConfig(NamedTuple):
    r"""Configuration used by shared ProFITi forecasting-model tests."""

    input_dim: int
    latent_dim: int
    num_layers: int
    num_heads: int


class _TargetContext(nn.Module):
    r"""Minimal context encoder for ProFITi wrapper tests."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        *,
        timestamps: torch.Tensor,  # noqa: ARG002
        context_values: torch.Tensor,
        context_mask: torch.Tensor,  # noqa: ARG002
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        *batch_shape, _, _ = query_mask.shape
        max_targets = int(query_mask.sum(dim=(-2, -1)).max().item())
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
    x = torch.linspace(-5.0, 5.0, 101)

    y, forward_logabsdet = transform.encode_and_logabsdet(x)
    xhat, inverse_logabsdet = transform.decode_and_logabsdet(y)

    assert_close(xhat, x, atol=1e-5, rtol=1e-5)
    assert_close(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-5,
        rtol=1e-5,
    )


def test_shiesh_logabsdet_matches_autograd_derivative() -> None:
    r"""Check Shiesh logabsdet against autograd derivatives."""
    transform = Shiesh(t=1.0, a=1.0)
    x = torch.linspace(-5.0, 5.0, 101)

    _, actual = transform.encode_and_logabsdet(x)
    derivative = vmap(grad(lambda z: transform.encode_and_logabsdet(z)[0]))(x)
    expected = derivative.abs().log()

    assert actual.isfinite().all()
    assert expected.isfinite().all()
    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_triangular_attention_encode_decode_roundtrip_with_logabsdet() -> None:
    r"""Check that triangular attention encode/decode are inverse maps."""
    torch.manual_seed(0)
    attention = TriangularAttention(dim_context=4, dim_hidden=3)
    context = torch.randn(5, 4)
    x = torch.randn(5)

    y, forward_logabsdet = attention.encode_and_logabsdet(
        context,
        context,
        x.unsqueeze(-1),
    )
    xhat, inverse_logabsdet = attention.decode_and_logabsdet(context, context, y)
    xhat = xhat.squeeze(-1)

    assert_close(xhat, x, atol=1e-5, rtol=1e-5)
    assert_close(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-5,
        rtol=1e-5,
    )


def test_triangular_attention_logabsdet_matches_dense_jacobian() -> None:
    r"""Check triangular attention logabsdet against a dense Jacobian."""
    torch.manual_seed(0)
    num_steps = 5
    attention = TriangularAttention(dim_context=4, dim_hidden=3)
    context = torch.randn(num_steps, 4)
    x = torch.randn(num_steps)

    _, actual = attention.encode_and_logabsdet(context, context, x.unsqueeze(-1))

    def encode_flat(z: torch.Tensor, /) -> torch.Tensor:
        y, _ = attention.encode_and_logabsdet(
            context,
            context,
            z.reshape(num_steps, 1),
        )
        return y.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(encode_flat, x.reshape(-1))
    _, expected = torch.linalg.slogdet(jacobian)

    assert actual.isfinite()
    assert expected.isfinite()
    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_triangular_attention_padding_invariance_with_nan_padding() -> None:
    r"""Check masked NaN padding does not affect valid attention outputs."""
    torch.manual_seed(0)
    attention = TriangularAttention(dim_context=4, dim_hidden=3)
    context = torch.randn(5, 4)
    value = torch.randn(5, 1)

    DUMMY_VALUE = 123456.789

    expected, expected_logabsdet = attention.encode_and_logabsdet(
        context, context, value
    )

    padded_context = torch.cat([context, context.new_full((2, 4), torch.nan)])
    padded_value = torch.cat([value, value.new_full((2, 1), torch.nan)])

    attn_mask = padded_context.isfinite().any(dim=-1)

    padded_context = torch.where(attn_mask.unsqueeze(-1), padded_context, DUMMY_VALUE)
    padded_value = torch.where(attn_mask.unsqueeze(-1), padded_value, DUMMY_VALUE)

    padded_context = padded_context.requires_grad_()
    padded_value = padded_value.requires_grad_()

    actual, actual_logabsdet = attention.encode_and_logabsdet(
        padded_context,
        padded_context,
        padded_value,
        valid_mask=attn_mask,
    )

    assert_close(actual[:5], expected, atol=1e-5, rtol=1e-5)
    assert_close(actual_logabsdet, expected_logabsdet, atol=1e-5, rtol=1e-5)

    (actual.nansum() + actual_logabsdet.nansum()).backward()
    assert padded_context.grad is not None
    assert padded_value.grad is not None
    assert padded_context.grad.isfinite().all()
    assert padded_value.grad.isfinite().all()
    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()


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
    )
    context = torch.randn(num_steps, latent_dim)
    x = torch.randn(num_steps)

    y, forward_logabsdet = flow.encode_and_logabsdet(x, context)
    xhat, inverse_logabsdet = flow.decode_and_logabsdet(y, context)

    assert_close(xhat, x, atol=1e-5, rtol=1e-5)
    assert_close(
        forward_logabsdet + inverse_logabsdet,
        torch.zeros_like(forward_logabsdet),
        atol=1e-5,
        rtol=1e-5,
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
    )
    context = torch.randn(num_steps, latent_dim)
    x = torch.randn(num_steps)

    _, actual = flow.encode_and_logabsdet(x, context)

    def encode_flat(z: torch.Tensor, /) -> torch.Tensor:
        y, _ = flow.encode_and_logabsdet(z.reshape(num_steps), context)
        return y.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(encode_flat, x.reshape(-1))
    _, expected = torch.linalg.slogdet(jacobian)

    assert actual.isfinite()
    assert expected.isfinite()
    assert_close(actual, expected, atol=1e-4, rtol=1e-4)


class TestProFITiBlock:
    r"""Tests for ProFITi block transforms."""

    def test_encode_decode_roundtrip_with_logabsdet(self) -> None:
        r"""Check that a ProFITi block encode/decode are inverse maps."""
        torch.manual_seed(0)
        num_steps = 5
        latent_dim = 4
        block = ProFITiBlock(latent_dim=latent_dim, num_layers=1)
        context = torch.randn(num_steps, latent_dim)
        x = torch.randn(num_steps)

        y, forward_logabsdet = block.encode_and_logabsdet(x, context)
        xhat, inverse_logabsdet = block.decode_and_logabsdet(y, context)

        assert_close(xhat, x, atol=1e-5, rtol=1e-5)
        assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_logabsdet_matches_dense_jacobian(self) -> None:
        r"""Check a ProFITi block logabsdet against a dense Jacobian."""
        torch.manual_seed(0)
        num_steps = 5
        latent_dim = 4
        block = ProFITiBlock(latent_dim=latent_dim, num_layers=1)
        context = torch.randn(num_steps, latent_dim)
        x = torch.randn(num_steps)

        _, actual = block.encode_and_logabsdet(x, context)

        def encode_flat(z: torch.Tensor, /) -> torch.Tensor:
            y, _ = block.encode_and_logabsdet(z.reshape(num_steps), context)
            return y.reshape(-1)

        jacobian = torch.autograd.functional.jacobian(encode_flat, x.reshape(-1))
        _, expected = torch.linalg.slogdet(jacobian)

        assert actual.isfinite()
        assert expected.isfinite()
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_variable_target_count_nan_padding_has_finite_gradients(self) -> None:
        r"""Check batches with NaN-padded target embeddings have finite gradients."""
        torch.manual_seed(0)
        max_targets = 5
        latent_dim = 4
        block = ProFITiBlock(latent_dim=latent_dim, num_layers=1)
        valid_mask = torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
            ]
        )
        raw_context = torch.randn(
            2,
            max_targets,
            latent_dim,
            requires_grad=True,
        )
        context = torch.where(
            valid_mask.unsqueeze(-1),
            raw_context,
            torch.nan,
        )
        z = torch.randn(2, max_targets, requires_grad=True)

        y, logabsdet = block.encode_and_logabsdet(z, context)
        loss = y.nansum() + logabsdet.nansum()
        loss.backward()

        assert y.shape == z.shape
        assert logabsdet.shape == (2,)

        for name, parameter in block.named_parameters():
            if parameter.grad is not None:
                assert parameter.grad.isfinite().all(), name

        assert z.grad is not None
        assert z.grad.isfinite().all()
        assert raw_context.grad is not None
        assert raw_context.grad.isfinite().all()


class TestProfiti(TestForecastingModel[ProFITi]):
    r"""Shared forecasting-model tests for ProFITi."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (4,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[ProFITiTestConfig] = ProFITiTestConfig(
        input_dim=CONTEXT_SHAPE[0],
        latent_dim=6,
        num_layers=2,
        num_heads=1,
    )

    @pytest.fixture
    def model_config(self) -> ProFITiTestConfig:
        r"""Configuration used to instantiate the ProFITi model under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> ProFITi:
        r"""Instantiate a ProFITi model from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, ProFITiTestConfig):
            raise TypeError("model_config must be a ProFITiTestConfig.")
        return ProFITi.from_config(
            ProFITiConfig(
                input_dim=model_config.input_dim,
                latent_dim=model_config.latent_dim,
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
            )
        )

    def forecast(
        self,
        model: ProFITi,
        inputs: SplitTimeData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return ProFITi target log densities for sequential forecasting inputs."""
        assert inputs.target_values is not None
        query_valid = inputs.query_mask.any(dim=-1)
        target_values = inputs.target_values

        log_prob = model.log_prob(
            target_values,
            query_times=inputs.query_times,
            query_mask=inputs.query_mask,
            context_times=inputs.context_times,
            context_mask=inputs.context_mask,
            context_values=inputs.context_values,
        )
        event_ndim = target_values.ndim - inputs.query_times.ndim
        log_prob = log_prob.reshape(*log_prob.shape, *((1,) * (event_ndim + 1)))
        predictions = torch.where(inputs.target_values.isfinite(), log_prob, torch.nan)

        assert predictions.shape == target_values.shape
        assert predictions[query_valid].isfinite().all()
        return (predictions,)

    def loss(
        self,
        model: ProFITi,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return ProFITi negative log likelihood for target values."""
        (log_prob,) = predictions
        regularizer = log_prob.new_zeros(())
        for parameter in model.parameters():
            regularizer = regularizer + parameter.sum()
        return -log_prob[targets.isfinite()].mean() + 1e-6 * regularizer

    def test_from_config_uses_grafiti_and_flow_sequence(self) -> None:
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

    def test_sample_and_log_prob_uses_standard_normal_latent(self) -> None:
        r"""Check ProFITi samples via the flow and returns transformed log-density."""
        torch.manual_seed(0)
        scale = 2.0
        model = ProFITi(
            context_embedding=_TargetContext(hidden_dim=3),
            conditional_flow=_ScaleDecodeFlow(scale=scale),
        )
        context_times = torch.tensor([0.0, 1.0])
        context_values = torch.tensor([[1.0, torch.nan], [2.0, 3.0]])
        context_mask = context_values.isfinite()
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
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )

        assert samples.shape == (5, *query_mask.shape)
        assert log_prob.shape == (5,)
        assert torch.equal(samples.isfinite(), query_mask.expand_as(samples))

        latents = samples[:, query_mask] / scale
        expected = -0.5 * (latents.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        expected = expected - query_mask.sum() * math.log(scale)
        assert_close(log_prob, expected)

        sample, log_prob = model.sample_and_log_prob(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )

        assert sample.shape == query_mask.shape
        assert log_prob.shape == ()
        assert torch.equal(sample.isfinite(), query_mask)

        latent = sample[query_mask] / scale
        expected = -0.5 * (latent.square() + math.log(2.0 * math.pi)).sum()
        expected = expected - query_mask.sum() * math.log(scale)
        assert_close(log_prob, expected)

    def test_sample_and_log_prob_handles_batch_dimensions(self) -> None:
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
        context_mask = context_values.isfinite()
        query_times = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        query_mask = torch.tensor(
            [
                [[True, False], [True, True], [False, True]],
                [[False, True], [True, False], [True, True]],
            ]
        )

        samples, log_prob = model.sample_and_log_prob(
            (2, 3),
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )

        assert samples.shape == (2, 3, *query_mask.shape)
        assert log_prob.shape == (2, 3, 2)
        assert torch.equal(samples.isfinite(), query_mask.expand_as(samples))

        latents = torch.stack(
            [
                samples[..., k, :, :][..., query_mask[k]] / scale
                for k in range(query_mask.shape[0])
            ],
            dim=-2,
        )
        expected = -0.5 * (latents.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        expected = expected - query_mask.sum(dim=(-2, -1)) * math.log(scale)
        assert_close(log_prob, expected)

    def test_log_prob_with_explicit_query_mask(self) -> None:
        r"""Check ProFITi log_prob evaluates correctly with an explicit query mask."""
        scale = 2.0
        model = ProFITi(
            context_embedding=_TargetContext(hidden_dim=3),
            conditional_flow=_ScaleDecodeFlow(scale=scale),
        )
        context_times = torch.tensor([0.0, 1.0])
        context_values = torch.tensor([[1.0, torch.nan], [2.0, 3.0]])
        context_mask = context_values.isfinite()
        query_times = torch.tensor([2.0, 3.0, 4.0])

        values = torch.tensor(
            [
                [1.0, torch.nan],
                [2.0, 3.0],
                [torch.nan, 4.0],
            ]
        )
        query_mask = values.isfinite()

        log_prob = model.log_prob(
            values,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )
        latent = values[query_mask] / scale
        expected = -0.5 * (latent.square() + math.log(2.0 * math.pi)).sum()
        expected = expected - query_mask.sum() * math.log(scale)
        assert log_prob.shape == ()
        assert_close(log_prob, expected)

    def test_variable_target_count_nan_padding_has_finite_gradients(self) -> None:
        r"""Check ProFITi handles batches with variable target counts."""
        torch.manual_seed(0)
        config = ProFITiConfig(
            input_dim=2,
            num_heads=1,
            latent_dim=4,
            num_layers=1,
        )
        model = ProFITi.from_config(config)
        context_times = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]],
        )
        context_values = torch.randn(
            2,
            2,
            config.input_dim,
            requires_grad=True,
        )
        context_mask = context_values.isfinite()
        query_times = torch.tensor(
            [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
        )
        query_mask = torch.tensor(
            [
                [[True, False], [True, True], [False, True]],
                [[False, True], [False, False], [True, False]],
            ]
        )

        samples, log_prob = model.sample_and_log_prob(
            3,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )
        value_log_prob = model.log_prob(
            samples[0].detach(),
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )
        loss = samples.nansum() + log_prob.nansum() + value_log_prob.nansum()
        loss.backward()

        assert samples.shape == (3, *query_mask.shape)
        assert log_prob.shape == (3, 2)
        assert value_log_prob.shape == (2,)
        assert torch.equal(samples.isfinite(), query_mask.expand_as(samples))
        assert value_log_prob.isfinite().all()
        assert context_values.grad is not None
        assert context_values.grad.isfinite().all()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert parameter.grad.isfinite().all(), name

    @pytest.mark.parametrize("size", [(), (5,), (1, 2, 3)])
    def test_sample_and_log_prob_consistent(self, size: tuple[int, ...]) -> None:
        r"""Check ProFITi sample log-probs match log_prob evaluated on the same samples."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        D = self.STANDARD_CONFIG.input_dim
        context_times = torch.tensor([0.0, 0.5, 1.0, 1.5])
        context_values = torch.randn(4, D)
        context_mask = torch.tensor([[True] * D, [True] * D, [False] * D, [True] * D])
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_times = torch.tensor([2.0, 2.5, 3.0, 3.5])
        query_mask = torch.zeros(4, D, dtype=torch.bool)
        query_mask[2] = True
        query_mask[3] = True

        samples, log_prob_direct = model.sample_and_log_prob(
            size,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )
        log_prob_via_sample = model.log_prob(
            samples,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_mask=context_mask,
            context_values=context_values,
        )

        assert_close(log_prob_direct, log_prob_via_sample)
