r"""Tests for Marginalizable Normalizing Flow forecasting."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import Tensor

from linodenet.forecasting.mnf import (
    MarginalizableNormalizingFlow,
    MixtureWeightsModel,
    MultiHeadAttention,
    SeparableEncoder,
)

from .base import SequentialData, TestForecastingModel


class MNFConfig(NamedTuple):
    r"""Configuration for a MarginalizableNormalizingFlow."""

    n_feats: int
    n_heads: int
    num_flow_layers: int
    num_bins: int = 8
    bounds: tuple[float, float] = (-5.0, 5.0)


class TestMNF(TestForecastingModel[MarginalizableNormalizingFlow]):
    r"""Tests for the MarginalizableNormalizingFlow density model."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
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

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> MarginalizableNormalizingFlow:
        r"""Instantiate an MNF from the given config."""
        if not isinstance(model_config, MNFConfig):
            raise TypeError(
                f"model_config must be an MNFConfig, got {type(model_config)}"
            )
        return MarginalizableNormalizingFlow(
            n_feats=model_config.n_feats,
            n_heads=model_config.n_heads,
            num_flow_layers=model_config.num_flow_layers,
            num_bins=model_config.num_bins,
            bounds=model_config.bounds,
        )

    def forecast(
        self, model: MarginalizableNormalizingFlow, inputs: SequentialData, /
    ) -> tuple[Tensor, ...]:
        r"""Return per-query-step log-probabilities under the MNF density.

        MNF is a static density model: it evaluates each query time step
        independently, ignoring temporal context.
        """
        target_values = inputs.target_values  # (*batch_shape, T, D)
        query_mask = inputs.query_valid  # (*batch_shape, T)
        batch_shape = target_values.shape[:-2]
        T = target_values.shape[-2]
        D = target_values.shape[-1]

        flat_values = target_values.reshape(-1, D)  # (B*T, D)
        flat_mask = query_mask.reshape(-1)  # (B*T,)

        log_prob_flat = flat_values.new_full((flat_values.shape[0],), torch.nan)
        if flat_mask.any():
            log_prob_flat[flat_mask] = model.log_prob(flat_values[flat_mask])

        log_prob = log_prob_flat.reshape(*batch_shape, T)  # (*batch_shape, T)
        # Expand to (*batch_shape, T, D) to match the output_shape convention used by
        # the base test class (query_axis = ndim - len(output_shape) - 1).
        return (log_prob.unsqueeze(-1).expand_as(target_values),)

    def loss(
        self,
        model: MarginalizableNormalizingFlow,
        predictions: tuple[Tensor, ...],
        targets: Tensor,
    ) -> Tensor:
        r"""Return mean negative log-likelihood over valid query positions."""
        del model, targets
        (log_prob_expanded,) = (
            predictions  # (*batch_shape, T, D) — all D slices identical
        )
        log_prob = log_prob_expanded[..., 0]  # (*batch_shape, T)
        valid = log_prob.isfinite()
        return -log_prob[valid].sum() / valid.sum()


class TestMHA:
    r"""Tests for the custom multi-head attention module."""

    Q_DIM: ClassVar[int] = 5
    K_DIM: ClassVar[int] = 7
    V_DIM: ClassVar[int] = 11
    DIM_HIDDEN: ClassVar[int] = 4
    DIM_OUTPUT: ClassVar[int] = 3
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
            dim_hidden=self.DIM_HIDDEN,
            dim_output=self.DIM_OUTPUT,
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
        r"""Forward pass should preserve batch axes and expose head outputs."""
        torch.manual_seed(0)
        model = self.make_model()
        q, k, v = self.make_inputs(batch_shape)

        actual = model(q, k, v)

        assert actual.shape == (
            *batch_shape,
            self.NUM_HEADS,
            self.QUERY_SIZE,
            self.DIM_OUTPUT,
        )
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

        masked = model(q, k, v, mask=mask)
        expected = masked.new_empty(masked.shape)
        for idx in range(q.shape[0]):
            expected[idx] = model(q[idx], k[idx, mask[idx]], v[idx, mask[idx]])

        torch.testing.assert_close(masked, expected)

    def test_nan_padded_queries_produce_nan_outputs_only_at_padded_queries(
        self,
    ) -> None:
        r"""NaN-padded query rows should stay localized to the corresponding outputs."""
        torch.manual_seed(0)
        model = self.make_model()
        q, k, v = self.make_inputs((2,))
        q[0, 2:] = torch.nan

        actual = model(q, k, v)  # (..., H, $Q, D)

        expected_finite = q.isfinite().all(dim=-1)
        actual_finite = actual.isfinite().all(dim=-3).all(dim=-1)

        assert torch.equal(actual_finite, expected_finite)

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


class TestSeparableEncoder:
    r"""Tests for the separable Moses encoder."""

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
        dim_input = (self.NUM_FREQUENCIES + 1) + self.NUM_CHANNELS + 1
        return SeparableEncoder(
            dim_input=dim_input,
            dim_hidden=self.DIM_HIDDEN,
            num_heads=2,
            num_components=self.NUM_COMPONENTS,
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
        r"""Backward pass should propagate finite gradients through valid entries."""
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
            if parameter.grad is not None:
                assert parameter.grad.isfinite().all()


class TestMixtureWeightsModel:
    def test_mixture_weights_model_returns_one_weight_vector_per_batch_element(
        self,
    ) -> None:
        r"""Each learned mixture query should produce a normalized weight vector."""
        torch.manual_seed(0)
        model = MixtureWeightsModel(
            num_components=3,
            num_heads=2,
            dim_input=5,
            dim_hidden=8,
        )
        embeddings = torch.randn(2, 4, 5)
        valid_mask = torch.tensor(
            [[True, True, False, True], [True, False, False, False]]
        )
        embeddings = embeddings.masked_fill(~valid_mask.unsqueeze(-1), torch.nan)

        weights = model(embeddings, valid_mask=valid_mask)

        assert weights.shape == (2, 3)
        assert weights.isfinite().all()
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(2, dtype=weights.dtype, device=weights.device),
        )

    def test_mixture_weights_model_returns_zero_for_fully_masked_sequences(
        self,
    ) -> None:
        r"""A fully padded sequence should still yield a finite normalized vector."""
        model = MixtureWeightsModel(
            num_components=2,
            num_heads=2,
            dim_input=4,
            dim_hidden=6,
        )
        embeddings = torch.full((1, 3, 4), torch.nan)
        valid_mask = embeddings.isfinite().all(dim=-1)

        weights = model(embeddings, valid_mask=valid_mask)

        assert weights.shape == (1, 2)
        assert weights.isfinite().all()
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(1, dtype=weights.dtype, device=weights.device),
        )
