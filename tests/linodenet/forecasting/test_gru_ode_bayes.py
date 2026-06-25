r"""Tests for GRU-ODE-Bayes forecasting."""

from typing import ClassVar

import pytest
import torch
from torch import nan

from linodenet.forecasting.gru_ode_bayes import (
    GRU_ODE,
    Decoder,
    GRU_Bayes,
    GRU_ODE_Bayes,
    GRUODEBayesConfig,
    ODE_Flow,
    TorchODESolver,
    gaussian_kl,
    gaussian_kl_logvar,
    update_masked,
)
from linodenet.forecasting.utils import BatchedForecastingRequest

from .base import SequentialData, TestForecastingModel


class TestGRU_ODE_Bayes(TestForecastingModel[GRU_ODE_Bayes]):
    r"""Tests for direct GRU-ODE-Bayes model construction."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (5,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[GRUODEBayesConfig] = GRUODEBayesConfig(
        input_size=CONTEXT_SHAPE[0],
        hidden_size=7,
        decoder_hidden_size=11,
        feature_embedding_size=3,
        bias=True,
        dropout_rate=0.0,
        step_size=0.1,
        solver="euler",
    )

    @pytest.fixture
    def model_config(self) -> GRUODEBayesConfig:
        r"""Configuration used to instantiate the GRU-ODE-Bayes model under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> GRU_ODE_Bayes:
        r"""Instantiate a GRU-ODE-Bayes model from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, GRUODEBayesConfig):
            raise TypeError("model_config must be a GRUODEBayesConfig.")
        config = model_config
        decoder = Decoder(
            config.input_size,
            config.hidden_size,
            config.decoder_hidden_size,
            bias=config.bias,
            dropout_rate=config.dropout_rate,
        )
        flow = ODE_Flow(
            GRU_ODE(config.hidden_size, bias=config.bias),
            TorchODESolver.new(config.solver, step_size=config.step_size),
        )
        update_cell = GRU_Bayes(
            config.input_size,
            config.hidden_size,
            config.feature_embedding_size,
            bias=config.bias,
        )
        return GRU_ODE_Bayes(
            config.input_size,
            config.hidden_size,
            decoder=decoder,
            flow=flow,
            update_cell=update_cell,
        )

    def make_cru(self) -> GRU_ODE_Bayes:
        r"""Instantiate a GRU-ODE-Bayes model from :attr:`STANDARD_CONFIG`."""
        return self.make_model(self.STANDARD_CONFIG)

    def forecast(
        self, model: GRU_ODE_Bayes, inputs: SequentialData, /
    ) -> tuple[torch.Tensor, ...]:
        r"""Return GRU-ODE-Bayes predictions for sequential forecasting inputs."""
        request = BatchedForecastingRequest(
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_values.isfinite(),
            query_times=inputs.query_times,
            query_mask=inputs.query_valid.unsqueeze(-1).expand_as(inputs.target_values),
            target_values=inputs.target_values,
        )
        assert request.target_values is not None

        log_prob = model.log_prob(
            request.target_values,
            context_times=request.context_times,
            context_values=request.context_values,
            context_mask=request.context_mask,
            query_times=request.query_times,
            query_mask=request.query_mask,
        )
        posterior_mean = model.pred_means
        posterior_logvar = model.pred_logvars
        query_steps = request.query_mask.any(dim=-1)
        query_mean = posterior_mean[query_steps]
        query_logvar = posterior_logvar[query_steps]
        query_log_prob = log_prob[query_steps]
        query_mean[~request.query_mask[query_steps]] = nan
        query_logvar[~request.query_mask[query_steps]] = nan
        pred_mean = torch.full_like(inputs.target_values, nan)
        pred_logvar = torch.full_like(inputs.target_values, nan)
        pred_log_prob = inputs.target_values.new_full(inputs.query_times.shape, nan)
        pred_mean[inputs.query_valid] = query_mean
        pred_logvar[inputs.query_valid] = query_logvar
        pred_log_prob[inputs.query_valid] = query_log_prob

        assert pred_mean.shape == inputs.target_values.shape
        assert pred_logvar.shape == inputs.target_values.shape
        assert pred_log_prob.shape == inputs.query_times.shape
        assert pred_mean[inputs.query_valid].isfinite().all()
        assert pred_logvar[inputs.query_valid].isfinite().all()
        assert pred_log_prob[inputs.query_valid].isfinite().all()
        return pred_mean, pred_logvar, pred_log_prob.unsqueeze(-1).expand_as(pred_mean)

    def loss(
        self,
        model: GRU_ODE_Bayes,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return GRU-ODE-Bayes negative log-likelihood for predictions."""
        del model
        _, _, pred_log_prob = predictions
        mask = targets.isfinite()
        return -pred_log_prob[..., 0][mask.any(dim=-1)].sum() / mask.sum()

    def test_instantiation(self) -> None:
        config = self.STANDARD_CONFIG
        model = self.make_cru()

        assert isinstance(model, GRU_ODE_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert isinstance(model.flow, ODE_Flow)
        assert isinstance(model.flow.vector_field, GRU_ODE)
        assert isinstance(model.flow.solver, TorchODESolver)
        assert model.flow.solver.step_size == config.step_size
        assert model.flow.vector_field.lin_hh.in_features == config.hidden_size
        assert isinstance(model.update_cell, GRU_Bayes)
        assert model.update_cell.input_size == config.input_size
        assert model.update_cell.hidden_size == config.hidden_size
        assert model.update_cell.feature_embedding_size == config.feature_embedding_size
        assert model.initial_state.shape == (config.hidden_size,)

    def test_instantiation_from_config(self) -> None:
        config = self.STANDARD_CONFIG

        model = GRU_ODE_Bayes.from_config(config)

        assert isinstance(model, GRU_ODE_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert isinstance(model.flow, ODE_Flow)
        assert isinstance(model.flow.solver, TorchODESolver)
        assert model.flow.solver.step_size == config.step_size
        assert isinstance(model.update_cell, GRU_Bayes)
        assert model.update_cell.feature_embedding_size == config.feature_embedding_size

    def test_instantiation_from_parameters(self) -> None:
        config = self.STANDARD_CONFIG

        model = GRU_ODE_Bayes.from_parameters(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            decoder_hidden_size=config.decoder_hidden_size,
            feature_embedding_size=config.feature_embedding_size,
            bias=config.bias,
            dropout_rate=config.dropout_rate,
            step_size=config.step_size,
            solver=config.solver,
        )

        assert isinstance(model, GRU_ODE_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert isinstance(model.flow, ODE_Flow)
        assert isinstance(model.flow.solver, TorchODESolver)
        assert model.flow.solver.step_size == config.step_size
        assert isinstance(model.update_cell, GRU_Bayes)
        assert model.update_cell.feature_embedding_size == config.feature_embedding_size

    @pytest.mark.parametrize("size", [(), (5,), (1, 2, 3)])
    def test_sample_and_log_prob_consistent(self, size: tuple[int, ...]) -> None:
        torch.manual_seed(0)
        model = self.make_cru()
        D = self.STANDARD_CONFIG.input_size
        times = torch.tensor([0.0, 0.5, 1.0, 1.5])
        context_mask_all = torch.tensor(
            [[True] * D, [True] * D, [False] * D, [True] * D]
        )
        context_values_all = torch.randn(4, D).masked_fill(~context_mask_all, nan)
        query_mask_all = torch.zeros(4, D, dtype=torch.bool)
        query_mask_all[2] = True
        query_mask_all[3] = True

        ctx_steps = context_mask_all.any(dim=-1)  # [T, T, F, T]
        q_steps = query_mask_all.any(dim=-1)  # [F, F, T, T]
        context_times = times[ctx_steps]
        context_values = context_values_all[ctx_steps]
        context_mask = context_mask_all[ctx_steps]
        query_times = times[q_steps]
        query_mask = query_mask_all[q_steps]

        samples, log_prob_direct = model.sample_and_log_prob(
            size,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        log_prob_via_sample = model.log_prob(
            samples,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )

        torch.testing.assert_close(log_prob_direct, log_prob_via_sample)


class TestGRUODEBayes:
    r"""Tests for the cleaned GRU-ODE-Bayes implementation."""

    def test_forward_with_padded_partial_observations(self) -> None:
        torch.manual_seed(0)
        decoder = Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7)
        flow = ODE_Flow(
            GRU_ODE(5),
            TorchODESolver.new("euler", step_size=0.1),
        )
        update_cell = GRU_Bayes(
            input_size=3,
            hidden_size=5,
            feature_embedding_size=2,
        )
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=decoder,
            flow=flow,
            update_cell=update_cell,
        )

        context_times = torch.tensor([
            [0.0, 0.5, 1.0],
            [0.0, 0.25, nan],
        ])  # fmt: skip
        context_values = torch.randn(2, 3, 3)
        context_values[0, 1, 2] = nan
        context_values[1, 2] = nan
        context_mask = context_values.isfinite()
        query_times = torch.tensor([
            [1.2, 1.5],
            [0.5, nan],
        ])  # fmt: skip
        query_mask = query_times.isfinite().unsqueeze(-1).expand(2, 2, 3)

        combined = BatchedForecastingRequest(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
        ).to_combined()
        combined_step_mask = (combined.context_mask | combined.query_mask).any(dim=-1)

        posterior_mean, posterior_logvar = model.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )

        assert posterior_mean.shape == query_mask.shape
        assert posterior_logvar.shape == query_mask.shape
        assert posterior_mean[query_mask].isfinite().all()
        assert posterior_logvar[query_mask].isfinite().all()
        assert posterior_mean[~query_mask].isnan().all()
        assert posterior_logvar[~query_mask].isnan().all()

        assert model.prior_means.shape == combined.context_values.shape
        assert model.prior_logvars.shape == combined.context_values.shape
        assert model.post_means.shape == combined.context_values.shape
        assert model.post_logvars.shape == combined.context_values.shape
        assert model.prior_means[combined_step_mask].isfinite().all()
        assert model.post_logvars[combined_step_mask].isfinite().all()
        assert model.prior_means[~combined_step_mask].isnan().all()
        assert model.post_logvars[~combined_step_mask].isnan().all()

    def test_batch_first_false_matches_batch_first_true(self) -> None:
        r"""Check time-major inputs match batch-first inputs."""
        torch.manual_seed(0)
        batch_model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7),
            flow=ODE_Flow(
                GRU_ODE(5),
                TorchODESolver.new("euler", step_size=0.1),
            ),
            update_cell=GRU_Bayes(
                input_size=3,
                hidden_size=5,
                feature_embedding_size=2,
            ),
        )
        time_model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7),
            flow=ODE_Flow(
                GRU_ODE(5),
                TorchODESolver.new("euler", step_size=0.1),
            ),
            update_cell=GRU_Bayes(
                input_size=3,
                hidden_size=5,
                feature_embedding_size=2,
            ),
            batch_first=False,
        )
        time_model.load_state_dict(batch_model.state_dict())

        context_times = torch.tensor([[0.0, 0.5], [0.0, nan]])
        context_values = torch.randn(2, 2, 3)
        context_values[0, 1, 2] = nan
        context_values[1, 1] = nan
        context_mask = context_values.isfinite()
        query_times = torch.tensor([[0.75, 1.0], [0.25, nan]])
        query_mask = query_times.isfinite().unsqueeze(-1).expand(2, 2, 3)

        batch_mean, batch_logvar = batch_model.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        # batch_first=False model expects time-major inputs and returns (K, *batch, D)
        time_mean, time_logvar = time_model.predict(
            query_times=query_times.mT,
            query_mask=query_mask.moveaxis(-2, 0),
            context_times=context_times.mT,
            context_values=context_values.moveaxis(-2, 0),
            context_mask=context_mask.moveaxis(-2, 0),
        )

        torch.testing.assert_close(
            time_mean.moveaxis(0, -2), batch_mean, equal_nan=True
        )
        torch.testing.assert_close(
            time_logvar.moveaxis(0, -2), batch_logvar, equal_nan=True
        )
        torch.testing.assert_close(
            time_model.prior_means.moveaxis(0, -2),
            batch_model.prior_means,
            equal_nan=True,
        )
        torch.testing.assert_close(
            time_model.post_logvars.moveaxis(0, -2),
            batch_model.post_logvars,
            equal_nan=True,
        )

    def test_log_prob_returns_time_marginal_likelihoods(self) -> None:
        r"""Check GRU-ODE-Bayes log-probabilities are returned per time step."""
        torch.manual_seed(0)
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7),
            flow=ODE_Flow(
                GRU_ODE(5),
                TorchODESolver.new("euler", step_size=0.1),
            ),
            update_cell=GRU_Bayes(
                input_size=3,
                hidden_size=5,
                feature_embedding_size=2,
            ),
        )
        # context: t=0.0 (ch 0,1), t=0.5 (ch 0,2), t=1.5 (ch 0,1,2)
        context_times = torch.tensor([0.0, 0.5, 1.5])
        context_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, True],
                [True, True, True],
            ]
        )
        context_values = torch.randn(3, 3).masked_fill(~context_mask, nan)
        # query: t=0.5 (ch 1), t=1.0 (ch 0,2)
        query_times = torch.tensor([0.5, 1.0])
        query_mask = torch.tensor(
            [
                [False, True, False],
                [True, False, True],
            ]
        )
        values = torch.randn(2, 3).masked_fill(~query_mask, nan)

        log_prob = model.log_prob(
            values,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        mean, logvar = model.predict(
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        normal = torch.distributions.Normal(mean, torch.exp(0.5 * logvar))
        expected = torch.where(
            query_mask,
            normal.log_prob(values.nan_to_num(0.0)),
            0.0,
        ).sum(dim=-1)

        assert log_prob.shape == query_times.shape
        torch.testing.assert_close(log_prob, expected)

    def test_sample_returns_time_marginal_samples(self) -> None:
        r"""Check GRU-ODE-Bayes samples have the requested sample and query shape."""
        torch.manual_seed(0)
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7),
            flow=ODE_Flow(
                GRU_ODE(5),
                TorchODESolver.new("euler", step_size=0.1),
            ),
            update_cell=GRU_Bayes(
                input_size=3,
                hidden_size=5,
                feature_embedding_size=2,
            ),
        )
        # context: t=0.0 (ch 0,2), t=1.0 (ch 0,1)
        context_times = torch.tensor([0.0, 1.0])
        context_mask = torch.tensor(
            [
                [True, False, True],
                [True, True, False],
            ]
        )
        context_values = torch.randn(2, 3).masked_fill(~context_mask, nan)
        # query: t=0.0 (ch 1), t=0.5 (ch 0,2)
        query_times = torch.tensor([0.0, 0.5])
        query_mask = torch.tensor(
            [
                [False, True, False],
                [True, False, True],
            ]
        )

        samples = model.sample(
            5,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )

        assert samples.shape == (5, *query_mask.shape)
        assert samples[:, query_mask].isfinite().all()
        assert samples[:, ~query_mask].isnan().all()

    def test_sample_and_log_prob_returns_time_marginal_likelihoods(self) -> None:
        r"""Check GRU-ODE-Bayes samples and log-probabilities use query masks."""
        torch.manual_seed(0)
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7),
            flow=ODE_Flow(
                GRU_ODE(5),
                TorchODESolver.new("euler", step_size=0.1),
            ),
            update_cell=GRU_Bayes(
                input_size=3,
                hidden_size=5,
                feature_embedding_size=2,
            ),
        )
        # batch 0: context at t=0.0,1.0; query at t=0.0,0.5
        # batch 1: context at t=0.0,0.25; query at t=0.25,0.75
        context_times = torch.tensor([[0.0, 1.0], [0.0, 0.25]])
        context_mask = torch.tensor(
            [
                [[True, False, True], [True, True, True]],
                [[True, True, False], [True, False, True]],
            ]
        )
        context_values = torch.randn(2, 2, 3).masked_fill(~context_mask, nan)
        query_times = torch.tensor([[0.0, 0.5], [0.25, 0.75]])
        query_mask = torch.tensor(
            [
                [[False, True, False], [True, False, True]],
                [[False, True, False], [True, True, False]],
            ]
        )

        samples, log_prob = model.sample_and_log_prob(
            (2, 3),
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )
        expected = model.log_prob(
            samples,
            query_times=query_times,
            query_mask=query_mask,
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
        )

        assert samples.shape == (2, 3, *query_mask.shape)
        assert log_prob.shape == (2, 3, *query_times.shape)
        assert samples[..., query_mask].isfinite().all()
        assert samples[..., ~query_mask].isnan().all()
        torch.testing.assert_close(log_prob, expected)

    def test_update_masked_update_state_with_empty_selection(self) -> None:
        r"""Check all-padding masks do not require forward-loop special cases."""
        torch.manual_seed(0)
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7),
            flow=ODE_Flow(
                GRU_ODE(5),
                TorchODESolver.new("euler", step_size=0.1),
            ),
            update_cell=GRU_Bayes(
                input_size=3,
                hidden_size=5,
                feature_embedding_size=2,
            ),
        )
        state = torch.randn(2, 5)
        observation = torch.full((2, 3), nan)
        mask = torch.zeros(2, dtype=torch.bool)

        result = update_masked(
            model.update_state,
            (state, observation),
            target=state,
            batch_mask=mask,
        )

        assert result.shape == state.shape
        torch.testing.assert_close(result, state)

    def test_gaussian_bayes_kl_logvar_matches_variance_formula(self) -> None:
        r"""Test the paper-side KL term against the direct variance formula."""
        mu_pre = torch.tensor([[0.0, 1.0, -1.0], [2.0, 0.0, 1.0]])
        var_pre = torch.tensor([[1.5, 0.4, 2.0], [0.8, 1.2, 0.7]])
        mu_post = torch.tensor([[0.2, 0.7, -0.8], [1.5, -0.1, 0.4]])
        var_post = torch.tensor([[1.1, 0.6, 1.4], [0.9, 1.5, 0.5]])
        mu_obs = torch.tensor([[0.4, nan, -1.2], [2.3, 0.2, 0.8]])
        var_obs = torch.tensor([[0.3, nan, 0.5], [0.2, 0.9, 0.4]])

        valid = mu_obs.isfinite() & var_obs.isfinite()
        mu_obs_clean = mu_obs.nan_to_num(0.0)
        var_obs_clean = var_obs.nan_to_num(1.0)

        var_bayes = var_pre * var_obs_clean / (var_pre + var_obs_clean)
        mu_bayes = (
            var_obs_clean / (var_pre + var_obs_clean) * mu_pre
            + var_pre / (var_pre + var_obs_clean) * mu_obs_clean
        )
        expected = torch.where(
            valid,
            gaussian_kl((mu_bayes, var_bayes), (mu_post, var_post)),
            0.0,
        ).sum(dim=-1)

        actual = GRU_ODE_Bayes.gaussian_bayes_kl_logvar(
            (mu_pre, var_pre.log()),
            (mu_post, var_post.log()),
            (mu_obs, var_obs.log()),
        )

        torch.testing.assert_close(actual, expected)

    def test_gaussian_kl_logvar_matches_variance_formula(self) -> None:
        r"""Test log-variance Gaussian KL against the variance-space helper."""
        mu_1 = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
        var_1 = torch.tensor([[0.5, 2.0], [1.2, 0.7]])
        mu_2 = torch.tensor([[0.4, 0.5], [1.5, -0.5]])
        var_2 = torch.tensor([[0.8, 1.5], [0.9, 2.5]])

        expected = gaussian_kl((mu_1, var_1), (mu_2, var_2))
        actual = gaussian_kl_logvar((mu_1, var_1.log()), (mu_2, var_2.log()))

        torch.testing.assert_close(actual, expected)
