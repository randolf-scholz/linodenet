r"""Tests for GRU-ODE-Bayes forecasting."""

from typing import ClassVar, NamedTuple

import torch
from torch.nn.utils.rnn import pad_sequence

from linodenet.forecasting.gru_ode_bayes import (
    GRU_ODE_Bayes,
    GRUODEBayesConfig,
    build_gru_ode_bayes,
    gaussian_kl,
    gaussian_kl_logvar,
)


class GRUODEBayesData(NamedTuple):
    r"""Random GRU-ODE-Bayes input/output data with original sequence lengths."""

    context_times: torch.Tensor
    context_values: torch.Tensor
    context_lengths: torch.Tensor
    query_times: torch.Tensor
    query_values: torch.Tensor
    query_lengths: torch.Tensor

    @property
    def context_mask(self) -> torch.Tensor:
        r"""Boolean mask for valid context sequence entries."""
        return (
            torch.arange(
                self.context_times.shape[-1], device=self.context_lengths.device
            )
            < self.context_lengths[..., None]
        )

    @property
    def query_mask(self) -> torch.Tensor:
        r"""Boolean mask for valid query sequence entries."""
        return (
            torch.arange(self.query_times.shape[-1], device=self.query_lengths.device)
            < self.query_lengths[..., None]
        )


class TestModel:
    r"""Tests for direct GRU-ODE-Bayes model construction."""

    STANDARD_CONFIG: ClassVar[GRUODEBayesConfig] = GRUODEBayesConfig(
        input_size=5,
        hidden_size=7,
        p_hidden=11,
        prep_hidden=3,
        bias=True,
        dropout_rate=0.0,
        step_size=0.1,
        solver="euler",
    )

    @classmethod
    def make_cru(cls) -> GRU_ODE_Bayes:
        r"""Instantiate a GRU-ODE-Bayes model from :attr:`STANDARD_CONFIG`."""
        config = cls.STANDARD_CONFIG
        return GRU_ODE_Bayes(
            config.input_size,
            config.hidden_size,
            config.p_hidden,
            prep_hidden=config.prep_hidden,
            bias=config.bias,
            dropout_rate=config.dropout_rate,
            step_size=config.step_size,
            solver=config.solver,
        )

    @classmethod
    def make_data(
        cls,
        *,
        seed: int,
        batch_shape: int | tuple[int, ...],
        min_steps: int,
        max_steps: int,
    ) -> GRUODEBayesData:
        r"""Sample random GRU-ODE-Bayes context and query data."""
        if min_steps < 1:
            raise ValueError("min_steps must be positive.")
        if max_steps < min_steps:
            raise ValueError("max_steps must be greater than or equal to min_steps.")

        config = cls.STANDARD_CONFIG
        generator = torch.Generator().manual_seed(seed)
        batch_shape = torch.Size(
            (batch_shape,) if isinstance(batch_shape, int) else batch_shape
        )
        if any(size < 1 for size in batch_shape):
            raise ValueError("batch_shape entries must be positive.")

        num_batches = max(batch_shape.numel(), 1)
        context_steps = torch.randint(
            min_steps, max_steps + 1, (num_batches,), generator=generator
        )
        query_steps = torch.randint(
            min_steps, max_steps + 1, (num_batches,), generator=generator
        )

        times = []
        values = []
        for steps in (context_steps, query_steps):
            time_sequences = [
                torch.sort(torch.rand(int(num_steps), generator=generator)).values
                for num_steps in steps
            ]
            value_sequences = [
                torch.randn(int(num_steps), config.input_size, generator=generator)
                for num_steps in steps
            ]

            if batch_shape == ():
                times.append(time_sequences[0])
                values.append(value_sequences[0])
            else:
                times.append(
                    pad_sequence(
                        time_sequences,
                        batch_first=True,
                        padding_value=torch.nan,
                    ).reshape(*batch_shape, -1)
                )
                values.append(
                    pad_sequence(
                        value_sequences,
                        batch_first=True,
                        padding_value=torch.nan,
                    ).reshape(*batch_shape, -1, config.input_size)
                )

        context_times, query_times = times
        context_values, query_values = values
        context_lengths = context_steps.reshape(batch_shape)
        query_lengths = query_steps.reshape(batch_shape)
        context_length = int(context_steps.max())
        query_length = int(query_steps.max())
        context_end_times = torch.take_along_dim(
            context_times,
            (context_lengths - 1).unsqueeze(-1),
            dim=-1,
        ).squeeze(-1)
        query_times = query_times + context_end_times[..., None]

        assert context_lengths.shape == batch_shape
        assert query_lengths.shape == batch_shape
        assert context_times.shape == (*batch_shape, context_length)
        assert context_values.shape == (*batch_shape, context_length, config.input_size)
        assert query_times.shape == (*batch_shape, query_length)
        assert query_values.shape == (*batch_shape, query_length, config.input_size)
        return GRUODEBayesData(
            context_times=context_times,
            context_values=context_values,
            context_lengths=context_lengths,
            query_times=query_times,
            query_values=query_values,
            query_lengths=query_lengths,
        )

    def test_instantiation(self) -> None:
        config = self.STANDARD_CONFIG
        model = self.make_cru()

        assert isinstance(model, GRU_ODE_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.step_size == config.step_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert model.vector_field.lin_hh.in_features == config.hidden_size
        assert model.gru_bayes.input_size == config.input_size
        assert model.gru_bayes.hidden_size == config.hidden_size
        assert model.gru_bayes.prep_hidden == config.prep_hidden
        assert model.initial_state.shape == (config.hidden_size,)

    def test_instantiation_from_config(self) -> None:
        config = self.STANDARD_CONFIG

        model = build_gru_ode_bayes(config)

        assert isinstance(model, GRU_ODE_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.step_size == config.step_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert model.gru_bayes.prep_hidden == config.prep_hidden

    def test_unbatched_forward(self) -> None:
        model = self.make_cru()
        data = self.make_data(seed=0, batch_shape=(), min_steps=2, max_steps=5)

        pred_mean, pred_logvar = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )

        assert pred_mean.shape == data.query_values.shape
        assert pred_logvar.shape == data.query_values.shape
        assert pred_mean.isfinite().all()
        assert pred_logvar.isfinite().all()

    def test_batched_forward(self) -> None:
        model = self.make_cru()
        data = self.make_data(seed=0, batch_shape=(4,), min_steps=2, max_steps=5)

        pred_mean, pred_logvar = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )

        assert pred_mean.shape == data.query_values.shape
        assert pred_logvar.shape == data.query_values.shape
        assert pred_mean[data.query_mask].isfinite().all()
        assert pred_logvar[data.query_mask].isfinite().all()
        assert pred_mean[~data.query_mask].isnan().all()
        assert pred_logvar[~data.query_mask].isnan().all()

    def test_training_unbatched(self) -> None:
        torch.manual_seed(0)
        model = self.make_cru()
        data = self.make_data(seed=0, batch_shape=(), min_steps=5, max_steps=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        pred_mean, pred_logvar = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )
        initial_loss = GRU_ODE_Bayes.nll_logvar(
            data.query_values, pred_mean, pred_logvar
        ).mean()

        for _ in range(3):
            optimizer.zero_grad()
            pred_mean, pred_logvar = model(
                data.query_times,
                data.context_times,
                data.context_values,
            )
            loss = GRU_ODE_Bayes.nll_logvar(
                data.query_values, pred_mean, pred_logvar
            ).mean()
            loss.backward()

            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                assert parameter.grad.abs().sum() > 0, name

            optimizer.step()

        pred_mean, pred_logvar = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )
        final_loss = GRU_ODE_Bayes.nll_logvar(
            data.query_values, pred_mean, pred_logvar
        ).mean()

        for name, parameter in model.named_parameters():
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss

    def test_training_batched(self) -> None:
        torch.manual_seed(0)
        model = self.make_cru()
        data = self.make_data(seed=0, batch_shape=(8,), min_steps=2, max_steps=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        pred_mean, pred_logvar = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )
        mask = data.query_mask.unsqueeze(-1).expand_as(data.query_values)
        initial_loss = GRU_ODE_Bayes.nll_logvar(
            data.query_values, pred_mean, pred_logvar, mask
        ).sum() / mask.sum()

        for _ in range(3):
            optimizer.zero_grad()
            pred_mean, pred_logvar = model(
                data.query_times,
                data.context_times,
                data.context_values,
            )
            loss = GRU_ODE_Bayes.nll_logvar(
                data.query_values, pred_mean, pred_logvar, mask
            ).sum() / mask.sum()
            loss.backward()

            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                assert parameter.grad.abs().sum() > 0, name

            optimizer.step()

        pred_mean, pred_logvar = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )
        final_loss = GRU_ODE_Bayes.nll_logvar(
            data.query_values, pred_mean, pred_logvar, mask
        ).sum() / mask.sum()

        for name, parameter in model.named_parameters():
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss


class TestGRUODEBayes:
    r"""Tests for the cleaned GRU-ODE-Bayes implementation."""

    def test_forward_with_padded_partial_observations(self) -> None:
        torch.manual_seed(0)
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            p_hidden=7,
            prep_hidden=2,
            step_size=0.1,
        )

        context_times = torch.tensor(
            [
                [0.0, 0.5, 1.0],
                [0.0, 0.25, torch.nan],
            ]
        )
        context_values = torch.randn(2, 3, 3)
        context_values[0, 1, 2] = torch.nan
        context_values[1, 2] = torch.nan
        query_times = torch.tensor(
            [
                [1.2, 1.5],
                [0.5, torch.nan],
            ]
        )

        pred_mean, pred_logvar = model(query_times, context_times, context_values)
        query_mask = query_times.isfinite()
        context_mask = context_times.isfinite()

        assert pred_mean.shape == (2, 2, 3)
        assert pred_logvar.shape == (2, 2, 3)
        assert pred_mean[query_mask].isfinite().all()
        assert pred_logvar[query_mask].isfinite().all()
        assert pred_mean[~query_mask].isnan().all()
        assert pred_logvar[~query_mask].isnan().all()

        assert model.prior_means.shape == (2, 3, 3)
        assert model.prior_logvars.shape == (2, 3, 3)
        assert model.posterior_means.shape == (2, 3, 3)
        assert model.posterior_logvars.shape == (2, 3, 3)
        assert model.prior_means[context_mask].isfinite().all()
        assert model.posterior_logvars[context_mask].isfinite().all()
        assert model.prior_means[~context_mask].isnan().all()
        assert model.posterior_logvars[~context_mask].isnan().all()

    def test_gaussian_bayes_kl_logvar_matches_variance_formula(self) -> None:
        r"""Test the paper-side KL term against the direct variance formula."""
        mu_pre = torch.tensor([[0.0, 1.0, -1.0], [2.0, 0.0, 1.0]])
        var_pre = torch.tensor([[1.5, 0.4, 2.0], [0.8, 1.2, 0.7]])
        mu_post = torch.tensor([[0.2, 0.7, -0.8], [1.5, -0.1, 0.4]])
        var_post = torch.tensor([[1.1, 0.6, 1.4], [0.9, 1.5, 0.5]])
        mu_obs = torch.tensor([[0.4, torch.nan, -1.2], [2.3, 0.2, 0.8]])
        var_obs = torch.tensor([[0.3, torch.nan, 0.5], [0.2, 0.9, 0.4]])

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
