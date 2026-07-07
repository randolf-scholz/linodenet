r"""Tests for Marginalizable Normalizing Flow components."""

import math
from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

from linodenet_models.mnf import (
    ConditionalGaussian,
    ConditionalLRS,
    ConditionalSplineFlow,
    LearnableLRS,
    MarginalizableNormalizingFlow,
    MixtureWeightsModel,
    Moses,
    MultiHeadAttention,
    SeparableEncoder,
)
from linodenet_models.utils import SplitTimeData

from .base import TestForecastingModel, make_forecasting_request


def _manual_mixture_samples(
    log_weights: Tensor, sample_shape: tuple[int, ...]
) -> Tensor:
    r"""Sample mixture indices with the same shape convention as `Categorical.sample`."""
    num_components = log_weights.shape[-1]
    flat_probs = log_weights.exp().reshape(-1, num_components)
    num_samples = math.prod(sample_shape) if sample_shape else 1
    flat_samples = torch.multinomial(
        flat_probs,
        num_samples=num_samples,
        replacement=True,
    )
    return flat_samples.mT.reshape((*sample_shape, *log_weights.shape[:-1]))


class MNFConfig(NamedTuple):
    r"""Configuration for a MarginalizableNormalizingFlow."""

    n_feats: int
    n_heads: int
    num_flow_layers: int
    num_bins: int = 8
    bounds: tuple[float, float] = (-5.0, 5.0)


class MosesTestConfig(NamedTuple):
    r"""Configuration for the Moses forecasting model tests."""

    input_dim: int
    latent_dim: int
    num_mixture_components: int
    num_flow_layers: int
    num_bins: int = 4
    bounds: tuple[float, float] = (-3.0, 3.0)
    num_encoder_heads: int = 2


class TestConditionalLRS:
    r"""Tests for the conditional rational linear spline transform."""

    DIM_CONTEXT: ClassVar[int] = 4
    NUM_BINS: ClassVar[int] = 8
    NUM_HEADS: ClassVar[int] = 2
    NUM_VALUES: ClassVar[int] = 5

    def make_layer(self) -> ConditionalLRS:
        r"""Instantiate the conditional spline layer under test."""
        return ConditionalLRS(
            self.DIM_CONTEXT,
            num_bins=self.NUM_BINS,
            num_heads=self.NUM_HEADS,
            x_bounds=(-3.0, 3.0),
            y_bounds=(-2.0, 4.0),
        )

    def make_flow(self, num_flow_layers: int = 2) -> ConditionalSplineFlow:
        r"""Instantiate a conditional spline flow under test."""
        return ConditionalSplineFlow(
            self.DIM_CONTEXT,
            num_heads=self.NUM_HEADS,
            num_flow_layers=num_flow_layers,
            num_bins=self.NUM_BINS,
            x_bounds=(-3.0, 3.0),
            y_bounds=(-2.0, 4.0),
        )

    def test_encode_decode_roundtrip_with_logabsdet(self) -> None:
        r"""Encode/decode should invert each other for a conditional spline layer."""
        torch.manual_seed(0)
        layer = self.make_layer()
        x = torch.randn(3, self.NUM_HEADS, self.NUM_VALUES)
        context = torch.randn(3, self.NUM_HEADS, self.NUM_VALUES, self.DIM_CONTEXT)

        y, forward_logabsdet = layer.encode_and_logabsdet(x, context)
        xhat, inverse_logabsdet = layer.decode_and_logabsdet(y, context)

        assert y.shape == x.shape
        assert forward_logabsdet.shape == x.shape[:-1]
        assert_close(xhat, x, atol=1e-5, rtol=1e-5)
        assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_flow_roundtrip_with_logabsdet(self) -> None:
        r"""A chain of conditional spline layers should remain invertible."""
        torch.manual_seed(0)
        flow = self.make_flow(num_flow_layers=2)
        x = torch.randn(3, self.NUM_HEADS, self.NUM_VALUES)
        context = torch.randn(3, self.NUM_HEADS, self.NUM_VALUES, self.DIM_CONTEXT)

        y, forward_logabsdet = flow.encode_and_logabsdet(x, context)
        xhat, inverse_logabsdet = flow.decode_and_logabsdet(y, context)

        assert_close(xhat, x, atol=1e-5, rtol=1e-5)
        assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=1e-5,
            rtol=1e-5,
        )


class TestLearnableLRS:
    r"""Tests for the unconditional rational linear spline transform."""

    NUM_BINS: ClassVar[int] = 8
    NUM_VALUES: ClassVar[int] = 5

    @pytest.mark.parametrize(
        ("x_bounds", "y_bounds"),
        [
            ((-3.0, 3.0), (-3.0, 3.0)),
            ((-3.0, 3.0), (-2.0, 4.0)),
        ],
        ids=["identity", "affine"],
    )
    def test_initialization_is_globally_affine(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> None:
        r"""The default parameterization should start as a single affine map."""
        left, right = x_bounds
        bottom, top = y_bounds
        slope = (top - bottom) / (right - left)
        x_center = 0.5 * (left + right)
        y_center = 0.5 * (bottom + top)

        layer = LearnableLRS(
            num_heads=(),
            num_bins=self.NUM_BINS,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        x = torch.linspace(left + 0.25, right - 0.25, self.NUM_VALUES).repeat(3, 1)

        y, forward_logabsdet = layer.encode_and_logabsdet(x)
        xhat, inverse_logabsdet = layer.decode_and_logabsdet(y)

        expected = (x - x_center) * slope + y_center
        expected_logabsdet = torch.full_like(
            forward_logabsdet,
            math.log(slope),
        )

        assert_close(y, expected, atol=1e-6, rtol=1e-6)
        assert_close(forward_logabsdet, expected_logabsdet, atol=1e-6, rtol=1e-6)
        assert_close(xhat, x, atol=1e-6, rtol=1e-6)
        assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=1e-6,
            rtol=1e-6,
        )


class TestMarginalizableNormalizingFlow:
    r"""Tests for the unconditional MarginalizableNormalizingFlow density model."""

    NUM_FEATURES: ClassVar[int] = 3
    NUM_SAMPLES: ClassVar[int] = 5
    STANDARD_CONFIG: ClassVar[MNFConfig] = MNFConfig(
        n_feats=3,
        n_heads=4,
        num_flow_layers=2,
        num_bins=8,
    )

    @pytest.fixture
    def model_config(self) -> MNFConfig:
        r"""Configuration used to instantiate the MNF model under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(
        params=[(), (8,), (1, 2, 3)],
        ids=["batch_shape=()", "batch_shape=(8,)", "batch_shape=(1,2,3)"],
    )
    def batch_shape(self, request: pytest.FixtureRequest) -> tuple[int, ...]:
        r"""Batch shapes used to exercise broadcasted density evaluation."""
        return request.param

    def make_model(self, model_config: object, /) -> MarginalizableNormalizingFlow:
        r"""Instantiate an MNF from the given config."""
        if not isinstance(model_config, MNFConfig):
            raise TypeError(
                f"model_config must be an MNFConfig, got {type(model_config)}"
            )
        return MarginalizableNormalizingFlow(
            num_feats=model_config.n_feats,
            num_heads=model_config.n_heads,
            num_flow_layers=model_config.num_flow_layers,
            num_bins=model_config.num_bins,
            bounds=model_config.bounds,
        )

    def test_log_prob_returns_expected_shape(
        self,
        model_config: MNFConfig,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Log-probabilities should preserve all batch axes except the feature axis."""
        torch.manual_seed(0)
        model = self.make_model(model_config)
        target_values = torch.randn(*batch_shape, self.NUM_SAMPLES, self.NUM_FEATURES)

        log_prob = model.log_prob(target_values)

        assert log_prob.shape == (*batch_shape, self.NUM_SAMPLES)
        assert log_prob.isfinite().all()

    def test_log_prob_training_produces_finite_target_gradients(
        self,
        model_config: MNFConfig,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Training on unconditional targets should differentiate through values."""
        torch.manual_seed(0)
        model = self.make_model(model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        target_values = torch.randn(
            *batch_shape, self.NUM_SAMPLES, self.NUM_FEATURES
        ).requires_grad_()
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        initial_loss = -model.log_prob(target_values).mean()

        for _ in range(3):
            optimizer.zero_grad()
            target_values.grad = None
            loss = -model.log_prob(target_values).mean()
            loss.backward()

            assert target_values.grad is not None
            assert target_values.grad.isfinite().all()
            assert target_values.grad.abs().sum() > 0

            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                assert parameter.grad.abs().sum() > 0, name

            optimizer.step()

        final_loss = -model.log_prob(target_values).mean()

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss


class TestMoses(TestForecastingModel[Moses]):
    r"""Shared forecasting-model and smoke tests for Moses."""

    SEED: ClassVar[int] = 0
    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (2,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    GRADIENT_WARMUP_STEPS: ClassVar[int] = 1
    MIN_STEPS: ClassVar[int] = 2
    MAX_STEPS: ClassVar[int] = 4
    STANDARD_CONFIG: ClassVar[MosesTestConfig] = MosesTestConfig(
        input_dim=CONTEXT_SHAPE[0],
        latent_dim=8,
        num_mixture_components=2,
        num_flow_layers=1,
    )

    @pytest.fixture
    def model_config(self) -> MosesTestConfig:
        r"""Configuration used to instantiate the Moses model under test."""
        return self.STANDARD_CONFIG

    def make_model(self, model_config: object, /) -> Moses:
        r"""Instantiate a small Moses model for shared tests."""
        if not isinstance(model_config, MosesTestConfig):
            raise TypeError("model_config must be a MosesTestConfig.")
        return Moses(
            input_dim=model_config.input_dim,
            latent_dim=model_config.latent_dim,
            num_mixture_components=model_config.num_mixture_components,
            num_flow_layers=model_config.num_flow_layers,
            num_bins=model_config.num_bins,
            bounds=model_config.bounds,
            num_encoder_heads=model_config.num_encoder_heads,
        )

    def forecast(
        self,
        model: Moses,
        inputs: SplitTimeData,
        /,
    ) -> tuple[Tensor, ...]:
        r"""Return Moses joint log likelihood broadcast over valid target positions."""
        assert inputs.target_values is not None
        query_valid = inputs.query_mask.any(dim=-1)
        target_values = inputs.target_values

        log_prob = model.log_prob(
            target_values,
            query_times=inputs.query_times,
            query_mask=inputs.query_mask,
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_mask,
        )
        event_ndim = target_values.ndim - inputs.query_times.ndim
        log_prob = log_prob.reshape(*log_prob.shape, *((1,) * (event_ndim + 1)))
        predictions = torch.where(target_values.isfinite(), log_prob, torch.nan)

        assert predictions.shape == target_values.shape
        assert predictions[query_valid].isfinite().all()
        return (predictions,)

    def loss(
        self,
        model: Moses,  # noqa: ARG002
        predictions: tuple[Tensor, ...],
        targets: Tensor,
    ) -> Tensor:
        r"""Return Moses negative log likelihood for target values."""
        (log_prob,) = predictions
        return -log_prob[targets.isfinite()].mean()

    def make_inputs(self) -> dict[str, Tensor]:
        r"""Create a minimal irregular forecasting request."""
        data = make_forecasting_request(
            seed=self.SEED,
            batch_shape=(),
            min_steps=self.MIN_STEPS,
            max_steps=self.MAX_STEPS,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
            input_missingness=True,
            target_missingness=True,
        )
        assert data.target_values is not None

        return {
            "query_times": data.query_times.detach(),
            "query_mask": data.query_mask,
            "context_times": data.context_times.detach(),
            "context_values": data.context_values.detach(),
            "context_mask": data.context_mask,
            "values": data.target_values.detach(),
        }

    def test_initialization_succeeds(self) -> None:
        r"""A small Moses model should instantiate successfully."""
        model = self.make_model(self.STANDARD_CONFIG)

        assert isinstance(model, Moses)

    def test_sample_succeeds(self) -> None:
        r"""Sampling should succeed on a simple request."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        inputs = self.make_inputs()
        inputs.pop("values")

        samples = model.sample((), **inputs)

        assert samples.shape == inputs["query_mask"].shape
        assert torch.equal(samples.isfinite(), inputs["query_mask"])

    def test_log_prob_succeeds(self) -> None:
        r"""Density evaluation should succeed on a simple request."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        inputs = self.make_inputs()
        values = inputs.pop("values")

        log_prob = model.log_prob(values, **inputs)

        assert log_prob.shape == ()
        assert log_prob.isfinite()

    def test_sample_and_log_prob_consistent(self) -> None:
        r"""`sample_and_log_prob` should agree with `log_prob` on returned samples."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        inputs = self.make_inputs()
        inputs.pop("values")

        samples, log_prob_direct = model.sample_and_log_prob((), **inputs)
        log_prob_via_sample = model.log_prob(samples, **inputs)

        assert samples.shape == inputs["query_mask"].shape
        assert log_prob_direct.shape == ()
        assert torch.equal(samples.isfinite(), inputs["query_mask"])
        assert_close(log_prob_direct, log_prob_via_sample)


class TestMHA:
    r"""Tests for the custom multi-head attention module."""

    Q_DIM: ClassVar[int] = 5
    K_DIM: ClassVar[int] = 7
    V_DIM: ClassVar[int] = 11
    DIM_HEAD: ClassVar[int] = 4
    NUM_HEADS: ClassVar[int] = 2
    QUERY_SIZE: ClassVar[int] = 4
    KEY_SIZE: ClassVar[int] = 5

    @pytest.fixture(
        params=[(), (4,), (2, 3)],
        ids=["batch_shape=()", "batch_shape=(4,)", "batch_shape=(2,3)"],
    )
    def batch_shape(self, request: pytest.FixtureRequest) -> tuple[int, ...]:
        r"""Batch shapes used to exercise batched attention behavior."""
        return request.param

    def make_model(self) -> MultiHeadAttention:
        r"""Instantiate the attention module under test."""
        return MultiHeadAttention(
            self.Q_DIM,
            self.K_DIM,
            self.V_DIM,
            dim_head=self.DIM_HEAD,
            num_heads=self.NUM_HEADS,
        )

    def make_inputs(
        self, batch_shape: tuple[int, ...]
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Create random query/key/value tensors with the requested batch shape."""
        q = torch.randn(*batch_shape, self.QUERY_SIZE, self.Q_DIM)
        k = torch.randn(*batch_shape, self.KEY_SIZE, self.K_DIM)
        v = torch.randn(*batch_shape, self.KEY_SIZE, self.V_DIM)
        return q, k, v

    def test_forward_returns_expected_shape(self, batch_shape: tuple[int, ...]) -> None:
        r"""Forward pass should preserve batch axes and query width."""
        torch.manual_seed(0)
        model = self.make_model()
        q, k, v = self.make_inputs(batch_shape)

        actual = model(q, k, v)

        assert actual.shape == (*batch_shape, self.QUERY_SIZE, self.Q_DIM)
        assert actual.isfinite().all()

    def test_masked_forward_matches_truncated_inputs(self) -> None:
        r"""Masking invalid keys should match attending over the kept keys only."""
        torch.manual_seed(0)
        model = self.make_model()
        q, k, v = self.make_inputs((2,))
        mask = torch.tensor(
            [
                [True, False, True, False, True],
                [False, True, True, False, False],
            ]
        )

        masked = model(q, k, v, key_mask=mask)
        expected = masked.new_empty(masked.shape)
        for idx in range(q.shape[0]):
            expected[idx] = model(q[idx], k[idx, mask[idx]], v[idx, mask[idx]])

        assert_close(masked, expected)

    def test_nan_padded_queries_produce_nan_outputs_only_at_padded_queries(
        self,
    ) -> None:
        r"""NaN-padded query rows should stay localized to the corresponding outputs."""
        torch.manual_seed(0)
        model = self.make_model()
        q, k, v = self.make_inputs((2,))
        q[0, 2:] = torch.nan

        actual = model(q, k, v)  # (..., $Q, D)

        expected_finite = q.isfinite().all(dim=-1)
        actual_finite = actual.isfinite().all(dim=-1)

        assert torch.equal(actual_finite, expected_finite)

    def test_masked_nan_padded_queries_nansum_backward_keeps_parameter_gradients_finite(
        self,
    ) -> None:
        r"""Explicitly masked NaN-padded query rows should keep gradients finite."""
        torch.manual_seed(0)
        model = self.make_model()
        q, k, v = self.make_inputs((2,))
        q = q.requires_grad_()
        k = k.requires_grad_()
        v = v.requires_grad_()
        q.data[0, 2:] = torch.nan
        query_mask = q.isfinite().all(dim=-1)

        actual = model(q, k, v, query_mask=query_mask)
        actual.nansum().backward()

        for parameter in model.parameters():
            assert parameter.grad is not None
            assert parameter.grad.isfinite().all()

    def test_backward_produces_finite_gradients(
        self, batch_shape: tuple[int, ...]
    ) -> None:
        r"""Backward pass should yield finite gradients for inputs and parameters."""
        torch.manual_seed(0)
        model = self.make_model()
        q, k, v = self.make_inputs(batch_shape)
        q = q.requires_grad_()
        k = k.requires_grad_()
        v = v.requires_grad_()

        actual = model(q, k, v)
        loss = actual.square().sum()

        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.isfinite().all()
        assert k.grad.isfinite().all()
        assert v.grad.isfinite().all()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert parameter.grad.isfinite().all()


class TestConditionalGaussian:
    r"""Tests for the low-rank conditional Gaussian."""

    LATENT_DIM: ClassVar[int] = 4
    COVARIANCE_RANK: ClassVar[int] = 2
    NUM_HEADS: ClassVar[int] = 3
    NUM_QUERIES: ClassVar[int] = 5

    def make_model(self) -> ConditionalGaussian:
        r"""Instantiate the conditional Gaussian under test."""
        return ConditionalGaussian(
            latent_dim=self.LATENT_DIM,
            covariance_rank=self.COVARIANCE_RANK,
            num_heads=self.NUM_HEADS,
        )

    def test_init_requires_positive_covariance_rank(self) -> None:
        r"""The low-rank parameterization requires a strictly positive rank."""
        with pytest.raises(ValueError, match="covariance_rank must be positive"):
            ConditionalGaussian(
                latent_dim=self.LATENT_DIM,
                covariance_rank=0,
                num_heads=self.NUM_HEADS,
            )

    def test_cov_scale_returns_positive_floor(self) -> None:
        r"""The isotropic scale should stay strictly positive."""
        model = self.make_model()

        cov_scale = model.cov_scale()

        assert cov_scale.shape == (self.NUM_HEADS,)
        assert (cov_scale > model.cov_scale_min).all()

    def test_log_prob_matches_dense_multivariate_normal(self) -> None:
        r"""Woodbury log-probabilities should match the dense Gaussian formula."""
        torch.manual_seed(0)
        model = self.make_model()
        context = torch.randn(2, self.NUM_HEADS, self.NUM_QUERIES, self.LATENT_DIM)
        values = torch.randn(2, self.NUM_HEADS, self.NUM_QUERIES)

        mean, cov_factor, cov_scale = model.embed(context)
        cov_scale = cov_scale.to(dtype=context.dtype, device=context.device)
        covariance = cov_factor @ cov_factor.mT + cov_scale.square()[
            ..., None, None
        ] * torch.eye(self.NUM_QUERIES, dtype=context.dtype, device=context.device)
        reference = torch.distributions.MultivariateNormal(
            loc=mean, covariance_matrix=covariance
        )

        actual = model.log_prob(values, context)
        expected = reference.log_prob(values)

        assert actual.shape == (2, self.NUM_HEADS)
        assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_sample_matches_mean_and_covariance(self) -> None:
        r"""Samples should reproduce the implied first and second moments."""
        torch.manual_seed(0)
        model = ConditionalGaussian(
            latent_dim=self.LATENT_DIM,
            covariance_rank=self.COVARIANCE_RANK,
        )
        context = torch.randn(self.NUM_QUERIES, self.LATENT_DIM)

        mean, cov_factor, cov_scale = model.embed(context)
        cov_scale = cov_scale.to(dtype=context.dtype, device=context.device)
        expected_covariance = cov_factor @ cov_factor.mT + cov_scale.square() * (
            torch.eye(self.NUM_QUERIES, dtype=context.dtype, device=context.device)
        )

        samples = model.sample((8192,), context)
        sample_mean = samples.mean(dim=0)
        centered = samples - sample_mean
        sample_covariance = centered.mT @ centered / samples.shape[0]

        assert samples.shape == (8192, self.NUM_QUERIES)
        assert_close(sample_mean, mean, atol=0.06, rtol=0.06)
        assert_close(sample_covariance, expected_covariance, atol=0.10, rtol=0.10)


class TestSeparableEncoder:
    r"""Tests for the separable Moses encoder."""

    DIM_HEAD: ClassVar[int] = 4
    DIM_HIDDEN: ClassVar[int] = 8
    NUM_COMPONENTS: ClassVar[int] = 3
    NUM_FREQUENCIES: ClassVar[int] = 2
    NUM_CHANNELS: ClassVar[int] = 3
    CONTEXT_SIZE: ClassVar[int] = 5
    QUERY_SIZE: ClassVar[int] = 4
    CONTEXT_LENGTHS: ClassVar[tuple[int, ...]] = (3, 2, 4, 1)
    QUERY_LENGTHS: ClassVar[tuple[int, ...]] = (3, 1, 4, 2)

    @pytest.fixture(
        params=[(), (4,), (2, 3)],
        ids=["batch_shape=()", "batch_shape=(4,)", "batch_shape=(2,3)"],
    )
    def batch_shape(self, request: pytest.FixtureRequest) -> tuple[int, ...]:
        r"""Batch shapes used to exercise flatten/unflatten logic."""
        return request.param

    def make_model(self) -> SeparableEncoder:
        r"""Instantiate a separable encoder with scalar context values."""
        return SeparableEncoder(
            dim_output=self.DIM_HIDDEN,
            dim_head=self.DIM_HEAD,
            num_attn_heads=2,
            num_heads=self.NUM_COMPONENTS,
            num_frequencies=self.NUM_FREQUENCIES,
            num_channels=self.NUM_CHANNELS,
        )

    def make_inputs(self, batch_shape: tuple[int, ...]) -> dict[str, Tensor]:
        r"""Create padded triplet inputs with varied valid lengths per batch item."""
        batch_shape = torch.Size(batch_shape)
        num_batches = batch_shape.numel() or 1

        context_times = torch.full((num_batches, self.CONTEXT_SIZE), torch.nan)
        context_channels = torch.full(
            (num_batches, self.CONTEXT_SIZE), -1, dtype=torch.long
        )
        context_values = torch.full((num_batches, self.CONTEXT_SIZE), torch.nan)
        query_times = torch.full((num_batches, self.QUERY_SIZE), torch.nan)
        query_channels = torch.full(
            (num_batches, self.QUERY_SIZE), -1, dtype=torch.long
        )

        base_context_channels = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long)
        base_query_channels = torch.tensor([1, 0, 2, 1], dtype=torch.long)
        for idx in range(num_batches):
            context_len = self.CONTEXT_LENGTHS[idx % len(self.CONTEXT_LENGTHS)]
            query_len = self.QUERY_LENGTHS[idx % len(self.QUERY_LENGTHS)]
            offset = float(idx) / 10.0
            context_times[idx, :context_len] = (
                torch.arange(context_len, dtype=torch.float32) + offset
            )
            context_channels[idx, :context_len] = base_context_channels[:context_len]
            context_values[idx, :context_len] = (
                torch.arange(context_len, dtype=torch.float32) + 1.0 + idx
            )
            query_times[idx, :query_len] = (
                torch.arange(query_len, dtype=torch.float32) + 0.5 + offset
            )
            query_channels[idx, :query_len] = base_query_channels[:query_len]

        if batch_shape == ():
            return {
                "query_times": query_times[0],
                "query_channels": query_channels[0],
                "context_times": context_times[0],
                "context_channels": context_channels[0],
                "context_values": context_values[0],
            }

        return {
            "query_times": query_times.reshape(*batch_shape, self.QUERY_SIZE),
            "query_channels": query_channels.reshape(*batch_shape, self.QUERY_SIZE),
            "context_times": context_times.reshape(*batch_shape, self.CONTEXT_SIZE),
            "context_channels": context_channels.reshape(
                *batch_shape, self.CONTEXT_SIZE
            ),
            "context_values": context_values.reshape(*batch_shape, self.CONTEXT_SIZE),
        }

    def test_forward_returns_expected_shapes_and_nan_padding(
        self, batch_shape: tuple[int, ...]
    ) -> None:
        r"""Valid tokens should produce finite outputs and padded tokens NaNs."""
        model = self.make_model()
        inputs = self.make_inputs(batch_shape)

        h_obs, h_mix = model(**inputs)

        assert h_obs.shape == (*batch_shape, self.CONTEXT_SIZE, self.DIM_HIDDEN)
        assert h_mix.shape == (
            *batch_shape,
            self.NUM_COMPONENTS,
            self.QUERY_SIZE,
            self.DIM_HIDDEN,
        )

        context_valid = inputs["context_times"].isfinite()
        query_valid = inputs["query_times"].isfinite()
        h_mix_valid = (
            query_valid.unsqueeze(dim=-2)
            .unsqueeze(dim=-1)
            .expand(
                *batch_shape,
                self.NUM_COMPONENTS,
                self.QUERY_SIZE,
                self.DIM_HIDDEN,
            )
        )

        assert h_obs[context_valid].isfinite().all()
        assert h_obs[~context_valid].isnan().all()
        assert h_mix[h_mix_valid].isfinite().all()
        assert h_mix[~h_mix_valid].isnan().all()

    def test_backward_produces_finite_gradients(
        self, batch_shape: tuple[int, ...]
    ) -> None:
        r"""Backward pass should propagate finite gradients through NaN-safe paths."""
        torch.manual_seed(0)
        model = self.make_model()
        inputs = self.make_inputs(batch_shape)
        context_values = inputs["context_values"].clone().requires_grad_()
        inputs["context_values"] = context_values

        h_obs, h_mix = model(**inputs)
        loss = h_obs.nansum() + h_mix.nansum()

        loss.backward()

        assert context_values.grad is not None
        valid_context = context_values.isfinite()
        assert context_values.grad[valid_context].isfinite().all()
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None
            assert parameter.grad.isfinite().all()
            assert not parameter.grad.eq(0.0).all()

    def test_permutation_variance(self, batch_shape: tuple[int, ...]) -> None:
        r"""Query and context permutations should induce the expected symmetries."""
        torch.manual_seed(0)
        model = self.make_model()
        inputs = self.make_inputs(batch_shape)

        h_obs, h_mix = model(**inputs)

        query_perm = torch.randperm(self.QUERY_SIZE)
        context_perm = torch.randperm(self.CONTEXT_SIZE)

        query_permuted_inputs = {
            **inputs,
            "query_times": inputs["query_times"].index_select(-1, query_perm),
            "query_channels": inputs["query_channels"].index_select(-1, query_perm),
        }
        h_obs_query_perm, h_mix_query_perm = model(**query_permuted_inputs)

        assert_close(h_obs_query_perm, h_obs, equal_nan=True)
        assert_close(
            h_mix_query_perm,
            h_mix.index_select(-2, query_perm),
            equal_nan=True,
        )

        context_permuted_inputs = {
            **inputs,
            "context_times": inputs["context_times"].index_select(-1, context_perm),
            "context_channels": inputs["context_channels"].index_select(
                -1, context_perm
            ),
            "context_values": inputs["context_values"].index_select(-1, context_perm),
        }
        h_obs_context_perm, h_mix_context_perm = model(**context_permuted_inputs)

        assert_close(
            h_obs_context_perm,
            h_obs.index_select(-2, context_perm),
            equal_nan=True,
        )
        assert_close(h_mix_context_perm, h_mix, equal_nan=True)


class TestMixtureWeightsModel:
    @pytest.mark.parametrize(
        "batch_shape",
        [
            pytest.param((), id="batch_shape=()"),
            pytest.param((2,), id="batch_shape=(2,)"),
            pytest.param((2, 3), id="batch_shape=(2,3)"),
        ],
    )
    def test_mixture_weights_model_returns_one_weight_vector_per_batch_element(
        self,
        batch_shape: tuple[int, ...],
    ) -> None:
        r"""Each learned mixture query should produce a normalized weight vector."""
        torch.manual_seed(0)
        model = MixtureWeightsModel(
            num_components=3,
            num_attn_heads=2,
            dim_input=5,
            dim_hidden=8,
        )
        embeddings = torch.randn(*batch_shape, 4, 5)
        valid_mask = torch.rand(*batch_shape, 4) > 0.4
        embeddings = embeddings.masked_fill(~valid_mask.unsqueeze(-1), torch.nan)

        log_weights = model(embeddings, valid_mask=valid_mask)

        assert log_weights.shape == (*batch_shape, 3)
        assert log_weights.isfinite().all()
        assert_close(
            log_weights.logsumexp(dim=-1),
            torch.zeros_like(log_weights[..., 0]),
        )

    def test_mixture_weights_model_returns_zero_for_fully_masked_sequences(
        self,
    ) -> None:
        r"""A fully padded sequence should still yield a finite normalized log-vector."""
        model = MixtureWeightsModel(
            num_components=2,
            num_attn_heads=2,
            dim_input=4,
            dim_hidden=6,
        )
        embeddings = torch.full((1, 3, 4), torch.nan)
        valid_mask = embeddings.isfinite().all(dim=-1)

        log_weights = model(embeddings, valid_mask=valid_mask)

        assert log_weights.shape == (1, 2)
        assert log_weights.isfinite().all()
        assert_close(log_weights.logsumexp(dim=-1), log_weights.new_zeros((1,)))

    @pytest.mark.parametrize(
        ("batch_shape", "sample_shape"),
        [
            pytest.param((), (), id="batch_shape=(),sample_shape=()"),
            pytest.param((2,), (5,), id="batch_shape=(2,),sample_shape=(5,)"),
            pytest.param((2, 3), (4, 2), id="batch_shape=(2,3),sample_shape=(4,2)"),
        ],
    )
    def test_log_prob_matches_forward_gather(
        self,
        batch_shape: tuple[int, ...],
        sample_shape: tuple[int, ...],
    ) -> None:
        r"""Index log-probabilities should match the selected forward log-weights."""
        torch.manual_seed(0)
        model = MixtureWeightsModel(
            num_components=4,
            num_attn_heads=2,
            dim_input=5,
            dim_hidden=8,
        )
        embeddings = torch.randn(*batch_shape, 4, 5)
        valid_mask = torch.rand(*batch_shape, 4) > 0.2
        embeddings = embeddings.masked_fill(~valid_mask.unsqueeze(-1), torch.nan)

        log_weights = model(embeddings, valid_mask=valid_mask)
        indices = torch.randint(0, 4, (*sample_shape, *batch_shape))

        actual = model.log_prob(indices, embeddings, valid_mask)
        expected = log_weights.reshape(
            *([1] * len(sample_shape)),
            *log_weights.shape,
        ).expand(*indices.shape, 4)
        expected = expected.gather(dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)

        assert actual.shape == (*sample_shape, *batch_shape)
        assert_close(actual, expected)

    @pytest.mark.parametrize(
        ("batch_shape", "size", "expected_sample_shape"),
        [
            pytest.param((), (), (), id="batch_shape=(),size=()"),
            pytest.param((2,), 7, (7,), id="batch_shape=(2,),size=7"),
            pytest.param(
                (2, 3),
                (4, 2),
                (4, 2),
                id="batch_shape=(2,3),size=(4,2)",
            ),
        ],
    )
    def test_sample_returns_expected_shape_and_range(
        self,
        batch_shape: tuple[int, ...],
        size: int | tuple[int, ...],
        expected_sample_shape: tuple[int, ...],
    ) -> None:
        r"""Sampling should prepend sample axes and return valid component indices."""
        torch.manual_seed(0)
        model = MixtureWeightsModel(
            num_components=5,
            num_attn_heads=2,
            dim_input=6,
            dim_hidden=8,
        )
        embeddings = torch.randn(*batch_shape, 3, 6)
        valid_mask = torch.rand(*batch_shape, 3) > 0.3
        embeddings = embeddings.masked_fill(~valid_mask.unsqueeze(-1), torch.nan)
        log_weights = model(embeddings, valid_mask=valid_mask)

        torch.manual_seed(1234)
        actual = model.sample(size, embeddings=embeddings, valid_mask=valid_mask)
        torch.manual_seed(1234)
        expected = _manual_mixture_samples(log_weights, expected_sample_shape)

        assert actual.shape == (*expected_sample_shape, *batch_shape)
        assert actual.dtype == torch.long
        assert ((0 <= actual) & (actual < 5)).all()
        assert_close(actual, expected)

    @pytest.mark.parametrize(
        ("batch_shape", "size", "expected_sample_shape"),
        [
            pytest.param((), (), (), id="batch_shape=(),size=()"),
            pytest.param((2,), 6, (6,), id="batch_shape=(2,),size=6"),
            pytest.param(
                (2, 3),
                (4, 2),
                (4, 2),
                id="batch_shape=(2,3),size=(4,2)",
            ),
        ],
    )
    def test_sample_and_log_prob_is_self_consistent(
        self,
        batch_shape: tuple[int, ...],
        size: int | tuple[int, ...],
        expected_sample_shape: tuple[int, ...],
    ) -> None:
        r"""Joint sampling should return the sampled indices and their log-probabilities."""
        torch.manual_seed(0)
        model = MixtureWeightsModel(
            num_components=4,
            num_attn_heads=2,
            dim_input=5,
            dim_hidden=8,
        )
        embeddings = torch.randn(*batch_shape, 5, 5)
        valid_mask = torch.rand(*batch_shape, 5) > 0.25
        embeddings = embeddings.masked_fill(~valid_mask.unsqueeze(-1), torch.nan)
        log_weights = model(embeddings, valid_mask=valid_mask)

        torch.manual_seed(4321)
        samples, log_prob = model.sample_and_log_prob(
            size,
            embeddings=embeddings,
            valid_mask=valid_mask,
        )
        torch.manual_seed(4321)
        expected_samples = _manual_mixture_samples(log_weights, expected_sample_shape)

        assert samples.shape == (*expected_sample_shape, *batch_shape)
        assert log_prob.shape == (*expected_sample_shape, *batch_shape)
        assert_close(samples, expected_samples)
        assert_close(log_prob, model.log_prob(samples, embeddings, valid_mask))
