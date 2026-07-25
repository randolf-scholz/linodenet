r"""Generate and visualize Brownian trajectories with a random drift bifurcation."""

from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.lines import Line2D
from torch import Tensor, nan, nn

from linodenet.mappings.transforms.scalar import Sinh
from linodenet_models import (
    ContinuousTimeKalmanFilter,
    ContinuousTimeNKF,
    GRU_ODE_Bayes,
    Moses,
    ProbabilisticForecastingModel,
    ProFITi,
)
from linodenet_models.cru import CRUConfig, DecoderConfig, EncoderConfig, build_cru
from linodenet_models.linodenet_probabilistic import (
    KoopmanFilter,
    make_koopman_filter,
    make_linodenet_prob,
)
from linodenet_models.profiti import ProFITiConfig
from linodenet_models.utils import SplitTimeData
from tests.testing import PROJECT

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]
MODEL_NAMES = (
    # "LinodenetProbabilistic",
    # "KoopmanFilter",
    "Moses",
    # "ContinuousTimeNKF",
    # "CRU",
    "ProFITi",
    # "GRU_ODE_Bayes",
    # "ContinuousTimeKalmanFilter",
)


@dataclass(frozen=True, slots=True)
class BifurcationSampler:
    r"""Sample Brownian trajectories with a randomly signed drift after $t⁎$.

    The sampled values satisfy $Y(t) = σW(t) + Cβ(t - t⁎)₊$, where $W$ is a
    standard Brownian motion and $C ∈ {-1, 1}$ is sampled independently for
    every trajectory.
    """

    t_star: float
    beta: float
    sigma: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.t_star <= 1.0:
            msg = f"Expected t_star in [0, 1], got {self.t_star}."
            raise ValueError(msg)
        if self.beta < 0.0:
            msg = f"Expected beta to be non-negative, got {self.beta}."
            raise ValueError(msg)
        if self.sigma < 0.0:
            msg = f"Expected sigma to be non-negative, got {self.sigma}."
            raise ValueError(msg)

    def sample(
        self,
        num_trajectories: int,
        num_steps: int,
        /,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Return sorted times, trajectory values, and drift signs.

        Args:
            num_trajectories: Number of independent Brownian trajectories.
            num_steps: Number of randomly sampled time steps per trajectory.

        Returns:
            A triple $(T, Y, C)$ with shapes $(N, S)$, $(N, S)$, and $(N,)$.
        """
        if num_trajectories < 1:
            msg = f"Expected at least one trajectory, got {num_trajectories}."
            raise ValueError(msg)
        if num_steps < 1:
            msg = f"Expected at least one time step, got {num_steps}."
            raise ValueError(msg)

        times = (
            torch.rand(num_trajectories, num_steps, device=device).sort(dim=-1).values
        )
        time_deltas = torch.diff(times, dim=-1, prepend=torch.zeros_like(times[:, :1]))
        brownian_motion = torch.cumsum(
            self.sigma * torch.randn_like(times) * time_deltas.sqrt(), dim=-1
        )
        coin_result = (
            2
            * torch.randint(0, 2, (num_trajectories,), dtype=torch.int64, device=device)
            - 1
        )
        drift = coin_result[:, None] * self.beta * (times - self.t_star).clamp_min(0)
        return times, brownian_motion + drift, coin_result


def test_samples() -> None:
    r"""Visualize Brownian trajectories after their randomly signed bifurcation."""
    torch.manual_seed(0)
    sampler = BifurcationSampler(t_star=0.3, beta=2.0, sigma=0.2)
    times, values, coin_results = sampler.sample(100, 100)

    assert times.shape == values.shape == (100, 100)
    assert coin_results.shape == (100,)
    assert torch.all(times[:, 1:] >= times[:, :-1])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for time, value, coin_result in zip(times, values, coin_results, strict=True):
        color = "tab:blue" if coin_result.item() < 0 else "tab:orange"
        ax.plot(time, value, color=color, alpha=0.35, linewidth=0.8)

    ax.axvline(sampler.t_star, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="time", ylabel="value", xlim=(0, 1), title="Brownian-motion bifurcation"
    )
    ax.legend(
        handles=[
            Line2D([], [], color="tab:blue", label="negative drift"),
            Line2D([], [], color="tab:orange", label="positive drift"),
        ]
    )
    fig.savefig(RESULT_DIR / "bifurcation_samples.png", dpi=200)
    plt.close(fig)


def _make_model(model_name: str, /) -> nn.Module:
    r"""Construct a small univariate continuous-time model with a sample API."""
    match model_name:
        case "LinodenetProbabilistic":
            return make_linodenet_prob(
                input_size=1,
                state_update="forward",
                retention=0.6,
                decoder="shiesh",
            )
        case "KoopmanFilter":
            return make_koopman_filter(
                input_size=1,
                latent_size=3,
                decoder="lowrank",
                low_rank=1,
                n_iter=1,
            )
        case "Moses":
            return Moses(
                input_dim=1,
                latent_dim=64,
                num_mixture_components=8,
                num_flow_layers=2,
                num_bins=6,
                num_encoder_heads=2,
            )
        case "ContinuousTimeNKF":
            return ContinuousTimeNKF(
                1,
                4,
                decoder=Sinh(),
                system_matrix=0.1 * torch.randn(4, 4),
                observation_matrix=torch.randn(1, 4),
                process_noise=0.2,
                measurement_noise=0.5,
                initial_mean=torch.randn(4),
                initial_covariance=2.0,
                initial_state_learnable=True,
                process_noise_learnable=True,
                observation_noise_learnable=True,
            )
        case "CRU":
            return build_cru(
                CRUConfig(
                    input_size=1,
                    output_size=1,
                    latent_size=4,
                    encoder=EncoderConfig(input_size=1, output_size=4, hidden_size=8),
                    decoder=DecoderConfig(input_size=4, output_size=1, hidden_size=8),
                    num_basis=4,
                    bandwidth=1,
                    initial_variance=2.0,
                    validate_args=True,
                )
            )
        case "ProFITi":
            return ProFITi.from_config(
                ProFITiConfig(input_dim=1, latent_dim=8, num_layers=2, num_heads=1)
            )
        case "GRU_ODE_Bayes":
            return GRU_ODE_Bayes.from_parameters(
                input_size=1,
                hidden_size=8,
                decoder_hidden_size=8,
                feature_embedding_size=2,
                step_size=0.1,
                solver="euler",
            )
        case "ContinuousTimeKalmanFilter":
            return ContinuousTimeKalmanFilter(
                1,
                4,
                system_matrix=0.05 * torch.randn(4, 4),
                observation_matrix=torch.randn(1, 4),
                process_noise=0.2,
                measurement_noise=0.5,
                initial_mean=torch.randn(4),
                initial_covariance=2.0 * torch.eye(4),
                initial_state_learnable=True,
                process_noise_learnable=True,
                observation_noise_learnable=True,
            )
        case _:
            msg = f"Unknown model: {model_name}."
            raise ValueError(msg)


def _split_trajectories(times: Tensor, values: Tensor, t0: Tensor, /) -> SplitTimeData:
    r"""Split trajectories into NaN-padded context and targets at per-row times."""
    num_steps = times.shape[-1]
    positions = torch.arange(num_steps, device=times.device)
    context_lengths = (times < t0[:, None]).sum(dim=-1)
    query_lengths = num_steps - context_lengths
    context_valid = positions < context_lengths[:, None]
    query_valid = positions < query_lengths[:, None]
    query_indices = (context_lengths[:, None] + positions).clamp_max(num_steps - 1)

    context_times = times.masked_fill(~context_valid, nan)
    context_values = values.unsqueeze(-1).masked_fill(~context_valid[..., None], nan)
    query_times = times.gather(dim=-1, index=query_indices).masked_fill(
        ~query_valid, nan
    )
    target_values = (
        values.gather(dim=-1, index=query_indices)
        .unsqueeze(-1)
        .masked_fill(~query_valid[..., None], nan)
    )
    return SplitTimeData(
        context_times=context_times,
        context_values=context_values,
        context_mask=context_valid[..., None],
        query_times=query_times,
        query_mask=query_valid[..., None],
        target_values=target_values,
    )


def _negative_log_likelihood(model: nn.Module, data: SplitTimeData, /) -> Tensor:
    r"""Return the mean negative conditional log likelihood for valid targets."""
    assert data.target_values is not None
    probabilistic_model = cast("ProbabilisticForecastingModel", model)
    log_prob = probabilistic_model.log_prob(
        data.target_values,
        query_times=data.query_times,
        query_mask=data.query_mask,
        context_times=data.context_times,
        context_values=data.context_values,
        context_mask=data.context_mask,
    )
    return -log_prob.nanmean()


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_bifurcation_models(model_name: str) -> None:
    r"""Train sample-capable models and visualize their bifurcation predictions."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAINING_BATCH_SIZE = 32
    T_STAR = 0.3
    PATIENCE = 10
    NUM_TRAJECTORIES = 100
    NUM_STEPS = 100

    torch.manual_seed(0)
    sampler = BifurcationSampler(t_star=T_STAR, beta=2.0, sigma=0.2)
    times, values, coin_results = sampler.sample(
        NUM_TRAJECTORIES, NUM_STEPS, device=DEVICE
    )
    model = torch.compile(_make_model(model_name), fullgraph=True)
    model = model.to(device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    best_loss = torch.inf
    iterations_without_improvement = 0
    while iterations_without_improvement < PATIENCE:
        batch_indices = torch.randint(times.shape[0], (TRAINING_BATCH_SIZE,))
        batch_times = times[batch_indices]
        batch_values = values[batch_indices]
        data = _split_trajectories(
            batch_times,
            batch_values,
            torch.full_like(batch_times[:, 0], T_STAR),
        )
        optimizer.zero_grad()

        loss = -model.log_prob(
            data.target_values,
            query_times=data.query_times,
            query_mask=data.query_mask,
            context_times=data.context_times,
            context_values=data.context_values,
            context_mask=data.context_mask,
        ).nanmean()

        print(loss)
        assert loss.isfinite()
        loss.backward()
        optimizer.step()
        if loss < best_loss:
            best_loss = loss.detach()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

    model.eval()
    data = _split_trajectories(times, values, torch.full_like(times[:, 0], T_STAR))
    probabilistic_model = cast("ProbabilisticForecastingModel", model)
    with torch.no_grad():
        samples = (
            probabilistic_model.sample(
                (),
                query_times=data.query_times,
                query_mask=data.query_mask,
                context_times=data.context_times,
                context_values=data.context_values,
                context_mask=data.context_mask,
            )
            .squeeze(0)
            .cpu()
        )

    assert data.target_values is not None
    assert samples.shape == data.target_values.shape
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for context_time, context_value, query_time, sample, coin_result in zip(
        data.context_times.cpu(),
        data.context_values[..., 0].cpu(),
        data.query_times.cpu(),
        samples[..., 0],
        coin_results,
        strict=True,
    ):
        color = "tab:blue" if coin_result.item() < 0 else "tab:orange"
        ax.plot(context_time, context_value, color="0.5", alpha=0.2, linewidth=0.8)
        if model_name in {"Moses", "ProFITi"}:
            ax.plot(query_time, sample, color=color, alpha=0.35, linewidth=0.8)
        else:
            ax.scatter(query_time, sample, color=color, alpha=0.35, s=6)

    ax.axvline(T_STAR, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="time", ylabel="value", xlim=(0, 1), ylim=(-2, 2), title=model_name)
    ax.legend(
        handles=[
            Line2D([], [], color="tab:blue", label="negative ground-truth drift"),
            Line2D([], [], color="tab:orange", label="positive ground-truth drift"),
        ],
        loc=0,
    )
    fig.savefig(RESULT_DIR / f"{model_name}.png", dpi=200)
    plt.close(fig)
