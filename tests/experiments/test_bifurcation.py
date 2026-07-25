r"""Generate and visualize Brownian trajectories with a random drift bifurcation."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import matplotlib.pyplot as plt
import optuna
import pytest
import torch
from matplotlib.lines import Line2D
from torch import Tensor, nan, nn
from tqdm.auto import tqdm

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


class BifurcationDataset(TypedDict):
    r"""Independent bifurcation train/validation/test splits."""

    t_star: float
    train_times: Tensor
    train_values: Tensor
    train_data: SplitTimeData
    train_coin_results: Tensor
    validation_data: SplitTimeData
    validation_coin_results: Tensor
    test_data: SplitTimeData
    test_coin_results: Tensor


class LossHistory(TypedDict):
    r"""Tracked losses during bifurcation training."""

    step: list[int]
    train_batch: list[float]
    train: list[float]
    validation: list[float]
    test: list[float]


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
            device: Optional device on which samples are allocated.

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


def _make_model(model_name: str, model_config: Mapping[str, Any], /) -> nn.Module:
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
            config = {
                "input_dim": 1,
                "latent_dim": 64,
                "num_mixture_components": 8,
                "num_flow_layers": 2,
                "num_bins": 6,
                "num_encoder_heads": 2,
            } | dict(model_config)
            return Moses(**config)
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
            config = {
                "input_dim": 1,
                "latent_dim": 8,
                "num_layers": 2,
                "num_heads": 1,
            } | dict(model_config)
            return ProFITi.from_config(ProFITiConfig(**config))
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


def make_bifurcation_dataset(
    sampler: BifurcationSampler,
    /,
    *,
    num_train_trajectories: int,
    num_validation_trajectories: int,
    num_test_trajectories: int,
    num_steps: int,
    device: torch.device,
) -> BifurcationDataset:
    r"""Sample independent train/validation/test data for the bifurcation task."""
    train_times, train_values, train_coin_results = sampler.sample(
        num_train_trajectories,
        num_steps,
        device=device,
    )
    validation_times, validation_values, validation_coin_results = sampler.sample(
        num_validation_trajectories,
        num_steps,
        device=device,
    )
    test_times, test_values, test_coin_results = sampler.sample(
        num_test_trajectories,
        num_steps,
        device=device,
    )
    return {
        "t_star": sampler.t_star,
        "train_times": train_times,
        "train_values": train_values,
        "train_data": _split_trajectories(
            train_times,
            train_values,
            torch.full_like(train_times[:, 0], sampler.t_star),
        ),
        "train_coin_results": train_coin_results,
        "validation_data": _split_trajectories(
            validation_times,
            validation_values,
            torch.full_like(validation_times[:, 0], sampler.t_star),
        ),
        "validation_coin_results": validation_coin_results,
        "test_data": _split_trajectories(
            test_times,
            test_values,
            torch.full_like(test_times[:, 0], sampler.t_star),
        ),
        "test_coin_results": test_coin_results,
    }


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


def run_experiment(
    model_name: str,
    model_config: Mapping[str, Any],
    dataset: BifurcationDataset,
    /,
    *,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_steps: int = 200,
    validate_every: int = 1,
    patience: int = 10,
    compile_model: bool = True,
    model: nn.Module | None = None,
) -> tuple[nn.Module, LossHistory]:
    r"""Initialize and train a model, tracking train/validation/test NLL."""
    device = dataset["train_times"].device
    model = _make_model(model_name, model_config) if model is None else model
    if compile_model:
        model = torch.compile(model, fullgraph=True)
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: LossHistory = {
        "step": [],
        "train_batch": [],
        "train": [],
        "validation": [],
        "test": [],
    }
    best_validation_loss = float("inf")
    iterations_without_improvement = 0

    for step in range(1, max_steps + 1):
        model.train()
        batch_indices = torch.randint(
            dataset["train_times"].shape[0],
            (batch_size,),
            device=device,
        )
        batch_times = dataset["train_times"][batch_indices]
        batch_values = dataset["train_values"][batch_indices]
        batch_data = _split_trajectories(
            batch_times,
            batch_values,
            torch.full_like(batch_times[:, 0], dataset["t_star"]),
        )

        optimizer.zero_grad()
        batch_loss = _negative_log_likelihood(model, batch_data)
        if not batch_loss.isfinite():
            msg = f"Encountered non-finite training loss at {step=}."
            raise RuntimeError(msg)
        batch_loss.backward()
        optimizer.step()

        if step % validate_every == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                train_loss = _negative_log_likelihood(model, dataset["train_data"])
                validation_loss = _negative_log_likelihood(
                    model, dataset["validation_data"]
                )
                test_loss = _negative_log_likelihood(model, dataset["test_data"])
            if not (train_loss.isfinite() and validation_loss.isfinite()):
                msg = f"Encountered non-finite tracked loss at {step=}."
                raise RuntimeError(msg)
            if not test_loss.isfinite():
                msg = f"Encountered non-finite test loss at {step=}."
                raise RuntimeError(msg)

            train_value = float(train_loss)
            validation_value = float(validation_loss)
            test_value = float(test_loss)
            history["step"].append(step)
            history["train_batch"].append(float(batch_loss.detach()))
            history["train"].append(train_value)
            history["validation"].append(validation_value)
            history["test"].append(test_value)

            print(
                f"{model_name} {step=} "
                f"train={train_value:.4f} "
                f"validation={validation_value:.4f} "
                f"test={test_value:.4f}"
            )
            if validation_value < best_validation_loss:
                best_validation_loss = validation_value
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            if iterations_without_improvement >= patience:
                break

    model.eval()
    return model, history


def _initialize_model(
    model_name: str,
    trial: optuna.Trial,
    /,
) -> tuple[nn.Module, dict[str, Any]]:
    r"""Initialize a model with Optuna-suggested hyperparameters."""
    match model_name:
        case "Moses":
            bound = trial.suggest_float("bound", 2.0, 8.0)
            model_config = {
                "input_dim": 1,
                "latent_dim": trial.suggest_categorical(
                    "latent_dim",
                    [16, 32, 64, 128],
                ),
                "num_mixture_components": trial.suggest_categorical(
                    "num_mixture_components",
                    [2, 4, 8, 16],
                ),
                "num_flow_layers": trial.suggest_int("num_flow_layers", 1, 4),
                "num_bins": trial.suggest_categorical(
                    "num_bins",
                    [4, 6, 8, 12, 16],
                ),
                "bounds": (-bound, bound),
                "num_encoder_heads": trial.suggest_categorical(
                    "num_encoder_heads",
                    [1, 2, 4],
                ),
                "covariance_rank": trial.suggest_categorical(
                    "covariance_rank",
                    [None, 1, 2, 4, 8],
                ),
            }
            return Moses(**model_config), model_config
        case (
            "LinodenetProbabilistic"
            | "KoopmanFilter"
            | "ContinuousTimeNKF"
            | "CRU"
            | "ProFITi"
            | "GRU_ODE_Bayes"
            | "ContinuousTimeKalmanFilter"
        ):
            msg = f"Hyperparameter tuning is not implemented for {model_name}."
            raise NotImplementedError(msg)
        case _:
            msg = f"Unknown model: {model_name}."
            raise ValueError(msg)


def tune_model_hyperparameters(model_name: str, /) -> None:
    r"""Tune model hyperparameters on the bifurcation experiment with Optuna."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 0
    T_STAR = 0.3
    NUM_TRAIN_TRAJECTORIES = 256
    NUM_VALID_TRAJECTORIES = 128
    NUM_TEST_TRAJECTORIES = 128
    NUM_STEPS = 100
    MAX_TRAIN_STEPS = 200
    VALIDATE_EVERY = 25
    PATIENCE = 8
    TIMEOUT = 1800
    N_TRIALS = 64

    torch.manual_seed(SEED)
    sampler = BifurcationSampler(t_star=T_STAR, beta=2.0, sigma=0.2)
    dataset = make_bifurcation_dataset(
        sampler,
        num_train_trajectories=NUM_TRAIN_TRAJECTORIES,
        num_validation_trajectories=NUM_VALID_TRAJECTORIES,
        num_test_trajectories=NUM_TEST_TRAJECTORIES,
        num_steps=NUM_STEPS,
        device=DEVICE,
    )

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(SEED + trial.number + 1)
        model, model_config = _initialize_model(model_name, trial)
        try:
            _, loss_history = run_experiment(
                model_name,
                model_config,
                dataset,
                batch_size=trial.suggest_categorical(
                    "train_batch_size",
                    [16, 32, 64],
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate",
                    1e-4,
                    3e-3,
                    log=True,
                ),
                weight_decay=trial.suggest_float(
                    "weight_decay",
                    1e-6,
                    1e-2,
                    log=True,
                ),
                max_steps=MAX_TRAIN_STEPS,
                validate_every=VALIDATE_EVERY,
                patience=PATIENCE,
                compile_model=False,
                model=model,
            )
            for step, validation_loss in zip(
                loss_history["step"],
                loss_history["validation"],
                strict=True,
            ):
                trial.report(validation_loss, step)
                if trial.should_prune():
                    raise optuna.TrialPruned
            trial.set_user_attr("loss_history", loss_history)
            return min(loss_history["validation"])
        except RuntimeError as exc:
            raise optuna.TrialPruned(str(exc)) from exc
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    config_dir = RESULT_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    model_stem = model_name.lower()
    storage_path = config_dir / f"{model_stem}_bifurcation_optuna.db"
    study = optuna.create_study(
        study_name=f"{model_stem}_bifurcation",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            multivariate=True,
            group=True,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=VALIDATE_EVERY,
        ),
    )

    try:
        study.optimize(
            objective, n_trials=N_TRIALS, timeout=TIMEOUT, gc_after_trial=True
        )
    finally:
        complete_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if complete_trials:
            best_params = study.best_params
            best_run_config = {
                "model_name": model_name,
                "model": {
                    "input_dim": 1,
                    "latent_dim": best_params["latent_dim"],
                    "num_mixture_components": best_params["num_mixture_components"],
                    "num_flow_layers": best_params["num_flow_layers"],
                    "num_bins": best_params["num_bins"],
                    "bounds": [-best_params["bound"], best_params["bound"]],
                    "num_encoder_heads": best_params["num_encoder_heads"],
                    "covariance_rank": best_params["covariance_rank"],
                },
                "optimizer": {
                    "name": "AdamW",
                    "lr": best_params["learning_rate"],
                    "weight_decay": best_params["weight_decay"],
                },
                "training": {
                    "train_batch_size": best_params["train_batch_size"],
                    "max_train_steps": MAX_TRAIN_STEPS,
                    "validate_every": VALIDATE_EVERY,
                    "patience": PATIENCE,
                },
                "data": {
                    "t_star": T_STAR,
                    "beta": sampler.beta,
                    "sigma": sampler.sigma,
                    "num_train_trajectories": NUM_TRAIN_TRAJECTORIES,
                    "num_valid_trajectories": NUM_VALID_TRAJECTORIES,
                    "num_test_trajectories": NUM_TEST_TRAJECTORIES,
                    "num_steps": NUM_STEPS,
                },
                "objective": {
                    "metric": "validation_nll",
                    "best_value": study.best_value,
                    "best_trial": study.best_trial.number,
                },
                "loss_history": study.best_trial.user_attrs.get("loss_history"),
            }
            (config_dir / f"{model_stem}_bifurcation_best_config.json").write_text(
                json.dumps(best_run_config, indent=2) + "\n",
                encoding="utf8",
            )

        study_summary = {
            "study_name": study.study_name,
            "storage": str(storage_path),
            "direction": study.direction.name,
            "num_trials": len(study.trials),
            "num_complete_trials": len(complete_trials),
            "best_value": study.best_value if complete_trials else None,
            "best_params": study.best_params if complete_trials else None,
            "trials": [
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": trial.params,
                    "intermediate_values": trial.intermediate_values,
                    "user_attrs": trial.user_attrs,
                }
                for trial in study.trials
            ],
        }
        (config_dir / f"{model_stem}_bifurcation_optuna_summary.json").write_text(
            json.dumps(study_summary, indent=2) + "\n",
            encoding="utf8",
        )

    assert complete_trials, "Optuna completed without any successful trials."
    assert torch.isfinite(torch.tensor(study.best_value))


@pytest.mark.slow
@pytest.mark.manual
def test_tune_moses_bifurcation_hyperparameters() -> None:
    r"""Tune Moses on the bifurcation experiment with Optuna.

    Run manually with:
        PYTHONPATH=src .venv/bin/python -m pytest -p no:rerunfailures --benchmark-skip \
            tests/experiments/test_bifurcation.py::test_tune_moses_bifurcation_hyperparameters
    """
    tune_model_hyperparameters("Moses")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_bifurcation_models(model_name: str) -> None:
    r"""Train sample-capable models and visualize their bifurcation predictions."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_STAR = 0.3
    NUM_TRAJECTORIES = 100
    NUM_STEPS = 100

    torch.manual_seed(0)
    sampler = BifurcationSampler(t_star=T_STAR, beta=2.0, sigma=0.2)
    dataset = make_bifurcation_dataset(
        sampler,
        num_train_trajectories=NUM_TRAJECTORIES,
        num_validation_trajectories=NUM_TRAJECTORIES,
        num_test_trajectories=NUM_TRAJECTORIES,
        num_steps=NUM_STEPS,
        device=DEVICE,
    )
    model, loss_history = run_experiment(model_name, {}, dataset)
    assert loss_history["validation"]

    data = dataset["test_data"]
    coin_results = dataset["test_coin_results"]
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


if __name__ == "__main__":
    test_tune_moses_bifurcation_hyperparameters()
